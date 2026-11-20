import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.vehicle import Vehicle
from utils.integrator import rk4_step_ca
from utils.quaternion import normalize_q, q_to_euler
from utils.animation import quaternion_to_rotation_matrix

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

n_failed = 0


def check(name, value, expected, tol):
    # compares value with expected (tolerance tol) and prints PASS or FAIL
    global n_failed
    ok = np.allclose(value, expected, atol=tol)
    if ok:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}: got {value}, expected {expected}")
        n_failed += 1


def make_state(u, p=(0, 0, 1000), v = (0, 0, 0), q = (1, 0, 0, 0), w = (0, 0, 0)):
    # builds a state vector from position, velocity, quaternion, angular velocity, and control inputs, default upright at rest, z = 1000 m
    return np.concatenate([p, v, q, w, u]).astype(float)

def simulate(vehicle, s0, u_dot, t_end, dt=0.01, normalize=True):
    # open loop: u_dot is kept const, RK4 steps of dt 
    f = lambda s, ud: np.array(vehicle.f(s, ud)).ravel() # casadi results -> plain numpy 
    s = s0.copy()
    states = [s]
    for _ in range(int(round(t_end / dt))):
        s = rk4_step_ca(f, s, u_dot, dt)
        if normalize:
            s = normalize_q(s)  # normalize quaternion
        states.append(s)
    t = np.arange(len(states)) * dt
    return t, np.array(states)


# where each part sits in the state vector
P, V, Q, W, U = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 18)
no_input = np.zeros(5) # u_dot = 0 -> actuators stay where they are



for snap in config['vehicle']['snapshots']:
    config['vehicle']['snapshot'] = snap
    veh = Vehicle(config)
    print(f"\n=== snapshot: {snap} (m = {veh.mass} kg, l_v = {veh.l_v} m) ===")


    # ------- check 1: hover -------
    d_h = veh.hover_throttle()
    print(f"hover throttle per engine: {d_h * 100:.1f} %")
    s0 = make_state(u=[d_h, d_h, 0, 0, 0])
    t, S = simulate(veh, s0, no_input, t_end=10.0)
    check("hover: position stays", S[-1, P], s0[P], tol=1e-6)
    check("hover: velocity stays 0", S[-1, V], 0, tol=1e-8)
    check("hover: no rotation", S[-1, W], 0, tol=1e-10)

    # ------- check 2: uneven throttle (engine 1 +5%, engine 2 -5%)
    s0 = make_state(u=[d_h + 0.05, d_h -0.05, 0, 0, 0])
    t, S = simulate(veh, s0, no_input, t_end=1)
    M_x = float(veh.tau_func(s0)[3])
    print(f" M_x from uneven throttle: {M_x:.1f} Nm")
    check("uneven: w_x after 1s = M_x / I_x", S[-1, W][0], M_x / veh.J[0, 0], tol=1e-6)
    check("uneven: w_y, w_z stay 0",           S[-1, W][1:], 0,                tol=1e-10)
    check("uneven: w_x > 0",                   S[-1, W][0] > 0, True,          tol=0)
    check("uneven: drifts to -y (v_y < 0)",    S[-1, V][1] < 0, True,          tol=0)
    check("uneven: no drift in x",             S[-1, P][0], 0,                 tol=1e-10)
    uneven = (t, S)

    # ------- check 3: thruster step on one axis
    for j, axis in enumerate(['x', 'y', 'z']):
        m = np.zeros(3)
        m[j] = 0.5                                     # 50 % RCS moment on axis j
        s0 = make_state(u=[d_h, d_h, *m])
        t, S = simulate(veh, s0, no_input, t_end=1)
        M_j = float(veh.tau_func(s0)[3 + j])
        others = [k for k in range(3) if k != j]
        check(f"rcs {axis}: w_{axis} after 1 s = M / I", S[-1, W][j], M_j / veh.J[j, j], tol=1e-6)
        check(f"rcs {axis}: other rates stay 0",        S[-1, W][others], 0,          tol=1e-10)

    # ------- check 4: u follows u_dot -------
    s0 = make_state(u=[d_h, d_h, 0, 0, 0])
    t, S = simulate(veh, s0, np.array([0.5, 0, 0, 0, 0]), t_end=0.2)
    check("ramp: delta_1 rises by 0.5 * 0.2 = 0.1", S[-1, U][0], d_h + 0.1, tol=1e-9)
    check("ramp: delta_2 unchanged",               S[-1, U][1], d_h,       tol=1e-12)

    # check 5: quaternion -------
    # tumbling on all axes, WITHOUT normalizing -> the length must stay 1 by itself
    s0 = make_state(u=[d_h, d_h, 0.2, -0.3, 0.1], w=(0.05, -0.02, 0.1))
    t, S = simulate(veh, s0, no_input, t_end=10, normalize=False)
    check("quaternion keeps unit length (10 s)", np.linalg.norm(S[:, Q], axis=1), 1, tol=1e-6)

    # pure yaw: 0.1 rad/s for 2 s -> yaw angle 0.2 rad
    s0 = make_state(u=[d_h, d_h, 0, 0, 0], w=(0, 0, 0.1))
    t, S = simulate(veh, s0, no_input, t_end=2)
    yaw = q_to_euler(S[-1:, Q])[0, 2]
    check("yaw 0.1 rad/s for 2 s -> 0.2 rad", yaw, 0.2, tol=1e-6)

    # -------- check 6: angular momentum is preserved in global cosi -------- 
    # no moment -> the angular momentum L = R(q) J w seen from the world must not change
    s0 = make_state(u=[d_h, d_h, 0, 0, 0], w=(0.3, -0.2, 0.5))
    t, S = simulate(veh, s0, no_input, t_end=20)
    L = [quaternion_to_rotation_matrix(s[Q]) @ veh.J @ s[W] for s in S]
    check("no moment: angular momentum constant (world)", L[-1], L[0], tol=1e-3)


# ------- summary -------
print()
if n_failed == 0:
    print("all checks passed")
else:
    print(f"{n_failed} check(s) FAILED")


# --- plot: uneven throttle (last snapshot) -------------------------------
t, S = uneven
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
ax[0].plot(t, S[:, 10])
ax[0].set_ylabel('w_x [rad/s]')
ax[1].plot(t, S[:, 4])
ax[1].set_ylabel('v_y [m/s]')
ax[2].plot(t, S[:, 1])
ax[2].set_ylabel('y [m]')
ax[2].set_xlabel('t [s]')
ax[0].set_title('uneven throttle: engine 1 +5 %, engine 2 -5 %')
for a in ax:
    a.grid()
plt.tight_layout()
plt.show()