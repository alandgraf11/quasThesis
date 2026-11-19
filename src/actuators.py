import numpy as np
import casadi as ca



def actuator_column(T, r, d):
    # one actuator (thrust T, position r, direction d) -> one 6x1 column of B
    M = ca.cross(r, d)           # moment = r x d
    return T * ca.vertcat(d, M)  # [force; moment], scaled by the thrust



def build_B(cfg, l_v):
    # column order = u = (delta_1, delta_2, m_x, m_y, m_z)
    T_max = cfg['T_max']
    l_l = cfg['l_l']
    beta = np.deg2rad(cfg['beta_deg'])
    s = ca.sin(beta)
    c = ca.cos(beta)

    # actuator positions
    r1 = ca.vertcat(0, l_l, -l_v)   # engine 1 position (at +l_l, below CoG)
    d1 = ca.vertcat(0, -c, s)       # engine 1 thrust direction
    r2 = ca.vertcat(0, -l_l, -l_v)  # engine 2 position (at -l_l)
    d2 = ca.vertcat(0, c, s)        # engine 2 thrust direction

    b1 = actuator_column(T_max, r1, d1)
    b2 = actuator_column(T_max, r2, d2)

    # rcs columns 
    
    rcs = cfg['rcs']
    rcs_cols = []

    # biulding the 3 rcs columns (one per axis)
    for j in range(3):                 # j = 0, 1, 2 -> x, y, z axis
        M_j = rcs['n_per_axis'] * rcs['F'] * rcs['lever'][j]
        col = ca.DM.zeros(6, 1)        # column of six zeros
        col[3 + j] = M_j               # put M_j in the moment row of axis j
        rcs_cols.append(col)

    # building the B matrix
    return ca.horzcat(b1, b2, *rcs_cols)   # 6 x 5 matrix

ROWS = ['F_x', 'F_y', 'F_z', 'M_x', 'M_y', 'M_z']
COLS = ['delta_1', 'delta_2', 'm_x', 'm_y', 'm_z']

# naming convention: F = force, M = moment, delta = throttle, m = rcs moment
def print_B(B):
    # prints B as a table with row and column names
    B = np.round(np.array(B), 1) + 0.0   # "+ 0.0" turns -0.0 into 0.0
    print('B in body frame (quas axes, z = thrust axis), units N / Nm')
    print(' ' * 6 + ''.join(f'{c:>10}' for c in COLS))
    for name, row in zip(ROWS, B):
        print(f'{name:<6}' + ''.join(f'{v:10.1f}' for v in row))




if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        print_B(build_B(config['propulsion'], 0.81))

