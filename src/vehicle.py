import numpy as np
import casadi as ca
from src.actuators import build_B
from utils.quaternion import euler_to_q


class Vehicle():
    def __init__(self, config, g=1.62):
        # pick mass and inertia snapshot from config (for stage 1 const per run)
        name = config['vehicle']['snapshot']
        snap = config['vehicle']['snapshots'][name]

        self.mass = snap['mass']          # kg
        self.J = np.diag([snap['Ix'], snap['Iy'], snap['Iz']])   # inertia matrix in body frame, kg m^2
        self.l_v = snap['l_v']                # m, distance from CoG to engine plane
        self.g = g                        # m/s^2, gravity acceleration

        # B from stap 1a: tau = B @ u 
        self.B = build_B(config['propulsion'], self.l_v)  # 6 x 5 matrix, body frame
        self.n_u = self.B.shape[1]  # number of control inputs (5)
        self.n_states = 13 + self.n_u  # number of states (13 + n_u = 18)

        # allowed actuator range U and rate limits (stage 1: boxes)
        prop = config['propulsion']
        d_min = prop['delta_min']
        rate = prop['rate_max']
        self.u_lb = np.array([d_min, d_min, -1, -1, -1])   # lower bounds of u
        self.u_ub = np.array([1, 1, 1, 1, 1])               # upper bounds of u
        self.u_dot_max = np.array([rate['engine'], rate['engine'], rate['rcs'], rate['rcs'], rate['rcs']])

        self.ca_model()

    def hover_throttle(self):
        # equal throttle for both engines to hover (no RCS)
        F_z_both = float(self.B[2,0] + self.B[2,1]) # total thrust in body z direction
        return self.mass * self.g / F_z_both  # throttle to hover (0..1)

    def make_state(self, rigid, u=None):
        # builds the full 18-state vector from the 12 rigid-body values in config.yaml:
        # rigid = [x, y, z, vx, vy, vz, phi, theta, psi (deg), wx, wy, wz]
        # u = actuators; default: both engines at hover throttle, thrusters off
        rigid = np.asarray(rigid, dtype=float)
        q = euler_to_q(np.deg2rad(rigid[6:9]))      # euler angles -> quaternion
        if u is None:
            d_h = self.hover_throttle()
            u = [d_h, d_h, 0, 0, 0]
        return np.concatenate([rigid[0:6], q, rigid[9:12], u])

    def ca_model(self):
        # ------------ states --------------
        p = ca.SX.sym('p', 3) # pos of CoG in world frame 
        v = ca.SX.sym('v', 3) # vel of CoG in world frame
        q = ca.SX.sym('q', 4) # quaternion of body frame in world frame
        w = ca.SX.sym('w', 3) # angular vel of body frame

        # ------------ controls --------------
        u = ca.SX.sym('u', self.n_u) # control inputs (2 engine throttles + 3 RCS moments)
        u_dot = ca.SX.sym('u_dot', self.n_u) # rate of change of the actuators

        # ----------- forces and moments -----------
                # ----------- fault parameters (stage 1d) -----------
        lam = ca.SX.sym('lam', self.n_u)   # effectiveness of each actuator: 1 = healthy, 0 = out
        b_f = ca.SX.sym('b_f', 6)          # constant extra force/moment (stuck actuator)

        tau = self.B @ (lam * u) + b_f     # tau = B Lambda_f u + b_f, in body frame
        F_body = tau[0:3]  # force in body frame
        M_body = tau[3:6]  # moment in body frame

        # rotation matrix body -> world frame
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = ca.vertcat(
            ca.horzcat(1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw),
            ca.horzcat(2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw),
            ca.horzcat(2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2)
        )

        # ----------- ODEs -----------
        g_W = ca.vertcat(0, 0, -self.g)

        p_dot = v
        v_dot = g_W + R @ F_body / self.mass

        wx, wy, wz = w[0], w[1], w[2]
        q_dot = 0.5 * ca.vertcat(
            -qx*wx - qy*wy - qz*wz,
            qw*wx + qy*wz - qz*wy,
            qw*wy + qz*wx - qx*wz,
            qw*wz + qx*wy - qy*wx
        )

        J = ca.DM(self.J)
        w_dot = ca.solve(J, M_body - ca.cross(w, J @ w)) # J w_dot = M - wx (J w) 

        # ----------- collect -----------
        s = ca.vertcat(p, v, q, w, u)  # state vector
        rhs = ca.vertcat(p_dot, v_dot, q_dot, w_dot, u_dot) # u is integrated from u_dot

        # dynamics and forces/moments WITH a fault (lam, b_f are extra inputs)
        self.f_fault = ca.Function('f_fault', [s, u_dot, lam, b_f], [rhs])
        self.tau_fault = ca.Function('tau_fault', [s, lam, b_f], [tau])

        # nominal versions (no fault: lam = 1, b_f = 0) -> same as before
        ones = ca.DM.ones(self.n_u)
        zeros = ca.DM.zeros(6)
        self.f = ca.Function('f', [s, u_dot], [self.f_fault(s, u_dot, ones, zeros)])
        self.tau_func = ca.Function('tau', [s], [self.tau_fault(s, ones, zeros)])