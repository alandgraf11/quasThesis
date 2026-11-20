import time
import numpy as np
import casadi as ca
from utils.integrator import rk4_step_ca
from src.faults import make_fault

class OCP_tracking(): # implements a finite horizon OCP with least squares cost (hence tracking)
    def __init__(self, config, vehicle):
        self.config = config
        self.vehicle = vehicle
        self.dynamics = vehicle.f   # rhs as a casadi function from vehicle.ca_model
        self.init_solver()

    def init_solver(self, print_level=0):
        
        dt = self.config['ocp']['time_step']    # time step size
        N = self.config['ocp']['horizon']       # prediction horizon   
        Q = ca.DM(np.diag(self.config['ocp']['Q']))   # state cost weights
        R = ca.DM(np.diag(self.config['ocp']['R']))   # control cost weights

        # casadi function of discrete dynamics
        n_states = self.dynamics.size1_in(0)
        n_controls = self.dynamics.size1_in(1)
        # n_outputs = self.dynamics.size1_out(0)

        s = ca.SX.sym('s', n_states, 1)
        ctrl = ca.SX.sym('u', n_controls, 1)
        s0 = ca.SX.sym('s0', n_states)
                # fault parameters: they enter the OCP as parameters -> no solver rebuild per fault
        lam = ca.SX.sym('lam', n_controls)
        b_f = ca.SX.sym('b_f', 6)
        f_fault = lambda x, u: self.vehicle.f_fault(x, u, lam, b_f)
        F = ca.Function('F', [s, ctrl, lam, b_f], [rk4_step_ca(f_fault, s, ctrl, dt)])

        # defining the cost (least squares)
        S = ca.SX.sym('S', n_states, N+1)
        U = ca.SX.sym('U', n_controls, N)
        S_goal = ca.SX.sym('s_goal', n_states, N+1) # the state the OCP is supposed to get to

        cost = 0
        for ii in range(N):
            e = S[:, ii] - S_goal[:, ii]
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[:, ii].T, R, U[:, ii]])
        e_N = S[:, N] - S_goal[:, N]
        cost += ca.mtimes([e_N.T, Q, e_N])

        # enforce the dynamics through constrains -> lbg and ubg
        g = []
        g.append(S[:, 0] - s0)

        for ii in range(N):
            g.append(S[6:10, ii+1].T @ S[6:10, ii+1] - 1)

            dyn_constraint = S[:, ii+1] - F(S[:, ii], U[:, ii], lam, b_f)
            g.append(dyn_constraint)

                # terminal constraint: position and velocity of the goal are reached exactly
        g.append(S[0:6, N] - S_goal[0:6, N])
        g = ca.vertcat(*g)
        lbg = ubg = np.zeros(g.shape[0])      

        # constraints on decision variables (states and controls) -> lbx and ubx
        s_lb = np.full(n_states, -np.inf)
        s_ub = np.full(n_states, np.inf)
        s_lb[2] = 0.0                      # ground: z >= 0
        s_lb[13:] = self.vehicle.u_lb      # actuators stay inside U
        s_ub[13:] = self.vehicle.u_ub
        u_lb = -self.vehicle.u_dot_max     # rate limits on the input u_dot
        u_ub = self.vehicle.u_dot_max       
        lbx = np.concatenate([np.tile(s_lb, N+1), np.tile(u_lb, N)])
        ubx = np.concatenate([np.tile(s_ub, N+1), np.tile(u_ub, N)])

        # collecting all decision variables in one vector
        # the vector is called x, because this is the way casadi specifies the 
        # vector of decision variables; don't confuse it with the position x, which is a scalar
        x = ca.vertcat(ca.vec(S), ca.vec(U))

        # vector of parameters
        # benefit of specifying them as parameters: solver does not need to be
        # instantiated every time that new parameters are specified
        p = ca.vertcat(s0, ca.vec(S_goal), lam, b_f)

        # setting up the nlp
        nlp = {'x': x,          # decision variables
                'f': cost,      # cost
                'g': g,         # constraints
                'p': p}         # parameters

        opts = {                # solver options
            'ipopt.print_level': print_level, # print_level=0 -> minimal; print_level=5 -> maximal
            'print_time': False,
            # 'error_on_fail': False,
        }

        # creating the solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.N = N
        self.dt = dt
        self.n_states = n_states
        self.n_controls = n_controls
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

    def solve(self, s0, s_goal, fault=None, x_init=None): # solving the OCP
        # faul: dict from src.faults.make_fault; None -> nominal
        # this function is vibe coded; don't worry too much about it, it's mostly ensuring
        # that arrays have the right dimension
        s0 = np.asarray(s0, dtype=float).ravel()
        s_goal = np.asarray(s_goal, dtype=float)    # s_goal can be defined for every time individually or identical for all time steps
        if s_goal.ndim == 1:                          
            s_goal = np.tile(s_goal, (self.N + 1, 1))
        assert s0.shape == (self.n_states,)
        assert s_goal.shape == (self.N + 1, self.n_states)

        if fault is None:
            fault = make_fault(self.vehicle)            # nominal
        p = np.concatenate([s0, s_goal.ravel(), fault['lam'], fault['b_f']])

        # rate limits of this fault (a slow actuator only changes these bounds)
        n_S = self.n_states * (self.N + 1)
        lbx = self.lbx.copy()
        ubx = self.ubx.copy()
        lbx[n_S:] = np.tile(-fault['u_dot_max'], self.N)
        ubx[n_S:] = np.tile(fault['u_dot_max'], self.N)

        if x_init is None:
            # initial guess: the lander stays at s0 for the whole horizon, u_dot = 0
            x_init = np.concatenate([np.tile(s0, self.N + 1),
                                    np.zeros(self.n_controls * self.N)])
        
        t0 = time.perf_counter()
        sol = self.solver(x0=x_init, p=p,               # solving the OCP with the provided intial state and parameters
                        lbx=lbx, ubx=ubx,
                        lbg=self.lbg, ubg=self.ubg)
        self.t_solve = time.perf_counter() - t0
        return sol

    def extract_sol(self, sol):         # extraing the state vector, control vector and stats from the solver
        x_opt = sol['x'].full().ravel()
        n_S = self.n_states * (self.N + 1)

        S_opt = x_opt[:n_S].reshape(self.N + 1, self.n_states)   # (N+1, N_states)
        U_opt = x_opt[n_S:].reshape(self.N, self.n_controls)     # (N,   N_controls)

        stats = self.solver.stats()
        info = {
            'success': stats['success'],
            'status': stats['return_status'],
            'iterations': stats.get('iter_count'),
            'solve_time': self.t_solve,
            'cost': float(sol['f']),
        }

        return S_opt, U_opt, info