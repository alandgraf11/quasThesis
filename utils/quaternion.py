import numpy as np

def euler_to_q(euler, state=None): # transform euler angles (rad) into quaternions
    def transform(phi, theta, psi):
        cr,sr = np.cos(phi/2),   np.sin(phi/2)
        cp,sp = np.cos(theta/2), np.sin(theta/2)
        cy,sy = np.cos(psi/2),   np.sin(psi/2)
        return np.array([cr*cp*cy + sr*sp*sy,
                            sr*cp*cy - cr*sp*sy,
                            cr*sp*cy + sr*cp*sy,
                            cr*cp*sy - sr*sp*cy])
    if euler is not None and state is None:
        phi, theta, psi = euler
        q = transform(phi, theta, psi)
        return q
    elif euler is None and state is not None:
        euler = state[6:9]
        q = euler_to_q(euler)
        state = np.concatenate((state[:6], q, state[9:]))
        return state

def normalize_q(state):
    state = np.array(state, dtype=float)   
    state[6:10] /= np.linalg.norm(state[6:10])
    return state

# def normalize_q(q, state=None): # normalize quaternion such that it's norm = 1
#     state=np.asarray(state)
#     print(state.shape)
#     def norm(q):
#         q = q / np.linalg.norm(q, axis=1, keepdims=True)
#         return q
#     if q is not None and state is None:
#         q = norm(q)
#         return q
#     elif q is None and state is not None:
#         q = state[6:10, 0]
#         q = normalize_q(q)
#         state[6:10, 0] = q 
#         return state

def q_to_euler(q): # transform quaternions to euler angles (rad)
    q = np.asarray(q)

    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected q with shape (N, 4), got {q.shape}")

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Roll (phi)
    phi = np.arctan2(2.0 * (w * x + y * z),
                    1.0 - 2.0 * (x * x + y * y))

    # Pitch (theta)
    sin_pitch = 2.0 * (w * y - z * x)

    # Numerical safety:
    # floating-point errors can occasionally produce
    # values slightly outside [-1, 1].
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)

    theta = np.arcsin(sin_pitch)

    # Yaw (psi)
    psi = np.arctan2(2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z))

    return np.column_stack((phi, theta, psi))