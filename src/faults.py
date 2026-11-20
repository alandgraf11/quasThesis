import numpy as np

ACTUATOR_NAMES = ['delta_1', 'delta_2', 'm_x', 'm_y', 'm_z']   # index 0 ... 4 in u


def make_fault(vehicle, kind='none', actuator=0, value=None):
    # a fault = (lam, b_f, u_dot_max): tau = B (lam * u) + b_f, |u_dot| <= u_dot_max
    #   kind      'none' | 'weak' | 'out' | 'stuck' | 'slow'
    #   actuator  index in u: 0 = delta_1, 1 = delta_2, 2 = m_x, 3 = m_y, 4 = m_z
    #   value     weak:  remaining effectiveness lam_i (0 ... 1)
    #             stuck: value the actuator is stuck at (e.g. throttle 0.4)
    #             slow:  remaining fraction of the rate limit (0 ... 1)

    lam = np.ones(vehicle.n_u)                  # 1 = healthy
    b_f = np.zeros(6)                           # no constant extra force / moment
    u_dot_max = vehicle.u_dot_max.copy()        # nominal rate limits
    i = actuator

    if kind == 'none':
        pass
    elif kind == 'weak':
        lam[i] = value
    elif kind == 'out':
        lam[i] = 0.0
    elif kind == 'stuck':
        lam[i] = 0.0 # commands no itput
        b_f = np.array(vehicle.B[:, i]).ravel() * value # const b_i * value stays instead
    elif kind == 'slow':
        u_dot_max[i] = u_dot_max[i] * value
    else: 
        raise ValueError(f"unknown fault kind '{kind}'")

    if kind == 'none':
        name = 'nominal' 
    else: 
        name = f"{ACTUATOR_NAMES[i]} {kind}" + (f" ({value})" if value is not None else "")

    return {'name': name, 'lam': lam, 'b_f': b_f, 'u_dot_max': u_dot_max,
            'kind': kind, 'actuator': actuator, 'value': value}
    