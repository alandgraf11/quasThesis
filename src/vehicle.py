import numpy as np
import casadi as ca
from src.actuators import build_B


class vehicle():
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

        self.ca_model()

    def hover_throttle(self):
        # equal throttle for both engines to hover (no RCS)
        F_z_both = float(self.B[2,0] + self.B[2,1]) # total thrust in body z direction
        return self.mass * self.g / F_z_both  # throttle to hover (0..1)

    def ca_model(self):
        # ------------ states --------------
        p = ca.SX.sym('p', 3) # pos of CoG in world frame 
        v = ca.SX.sym('v', 3) # vel of CoG in world frame
        q = ca.SX.sym('q', 4) # quaternion of body frame in world frame
        w = ca.SX.sym('w', 3) # angular vel of body frame

        # ------------ controls --------------
        u = ca.SX.sym('u', self.n_u) # control inputs (2 engine throttles + 3 RCS moments)

        # ----------- forces and moments -----------
        tau = self.B @ u  # tau = B @ u, in body frame
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

        self.f = ca.Function('f', [s, u_dot], [rhs]) # dynamics: s_dot = f(s, u_dot)
        self.tau_func = ca.Function('tau', [s], [tau]) # forces / moments for a given state s'