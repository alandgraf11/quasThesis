import casadi as ca

class Vehicle():
    def __init__(self, config, g=1.62):
        self.mass = config['vehicle']['mass']
        self.Ix = config['vehicle']['Ix']
        self.Iy = config['vehicle']['Iy']
        self.Iz = config['vehicle']['Iz']
        self.g = g

        self.ca_model()

    def ca_model(self): # dynamics of the vehicle build based upon casadi symbolic variables;
                        # can be used for numerical simulation and solving the OCP

        # states
        pos_W = ca.SX.sym('pos_W', 3, 1)            # world frame
        vel_W = ca.SX.sym('vel_W', 3, 1)            # world frame
        quat_B = ca.SX.sym('quat_B', 4, 1)          # body frame
        ang_vel_B = ca.SX.sym('ang_vel_B', 3, 1)    # body frame
        F_body = ca.SX.sym('F_body', 3, 1)          # Forces from the propulsion system are modelled as states (body frame)
        tau_body = ca.SX.sym('M_body', 3, 1)        # Torques from the propulsion system are modelled as states (body frame)
    
        x = pos_W[0]
        y = pos_W[1]
        z = pos_W[2]

        u = vel_W[0]
        v = vel_W[1]
        w = vel_W[2]

        qw = quat_B[0]
        qx = quat_B[1]
        qy = quat_B[2]
        qz = quat_B[3]

        p = ang_vel_B[0]
        q = ang_vel_B[1]
        r = ang_vel_B[2]

        Fx = F_body[0]
        Fy = F_body[1]
        Fz = F_body[2]

        taux = tau_body[0]
        tauy = tau_body[1]
        tauz = tau_body[2]

        # controls
        ctrl = ca.SX.sym('ctrl', 6, 1)

        # rate of change of the forces
        dFx = ctrl[0]
        dFy = ctrl[1]
        dFz = ctrl[2]
        
        # rate of change of the torques
        dtaux = ctrl[3]
        dtauy = ctrl[4]
        dtauz = ctrl[5]

        # rotation matrix from body to world frame; takes quaternions as inputs
        R = ca.vertcat(
            ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            ca.horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            ca.horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)),
        )

        F_world = R @ F_body

        # ODEs: translation in world frame, rotation in body frame
        xdot, ydot, zdot = u, v, w
        udot = F_world[0]/self.mass
        vdot = F_world[1]/self.mass
        wdot = F_world[2]/self.mass - self.g

        qwdot = -0.5*(qx*p + qy*q + qz*r)
        qxdot =  0.5*(qw*p + qy*r  - qz*q)
        qydot =  0.5*(qw*q + qz*p  - qx*r)
        qzdot =  0.5*(qw*r + qx*q - qy*p)

        pdot = (self.Iy-self.Iz)/self.Ix*q*r + taux/self.Ix
        qodot= (self.Iz-self.Ix)/self.Iy*r*p  + tauy/self.Iy
        rdot = (self.Ix-self.Iy)/self.Iz*p*q + tauz/self.Iz

        pos_dot = ca.vertcat(xdot, ydot, zdot)
        vel_dot = ca.vertcat(udot, vdot, wdot)
        quat_dot = ca.vertcat(qwdot, qxdot, qydot, qzdot)
        ang_vel_dot = ca.vertcat(pdot, qodot, rdot)

        # ODEs of the control inputs
        Fx_dot = dFx
        Fy_dot = dFy
        Fz_dot = dFz

        taux_dot = dtaux
        tauy_dot = dtauy
        tauz_dot = dtauz

        ctrls_dot = ca.vertcat(Fx_dot, Fy_dot, Fz_dot, taux_dot, tauy_dot, tauz_dot)

        # collecting all state derivatives in one vector
        rhs = ca.vertcat(pos_dot, vel_dot, quat_dot, ang_vel_dot, ctrls_dot)

        # state vector
        s = ca.vertcat(x, y, z, u, v, w,
                        qw, qx, qy, qz, p, q, r,
                        Fx, Fy, Fz, taux, tauy, tauz)
        
        # continous dynamics as casadi function
        self.f = ca.Function("f", [s, ctrl], [rhs])