import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.vehicle import Vehicle
from src.ocp import OCP_tracking
from utils.integrator import rk4_step_ca

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


def make_state(u, p=(0, 0, 0), v=(0, 0, 0), q=(1, 0, 0, 0), w=(0, 0, 0)):
    # builds the 18 states [p, v, q, w, u]
    return np.concatenate([p, v, q, w, u]).astype(float)


# where each part sits in the state vector
P, V, Q, W, U = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13), slice(13, 18)
results = {}


for snap in config['vehicle']['snapshots']:
    config['vehicle']['snapshot'] = snap
    veh = Vehicle(config)
    ocp = OCP_tracking(config, veh)
    print(f"\n=== snapshot: {snap}  (m = {veh.mass} kg) ===")

     # vertical landing: hover at 50 m at rest  ->  on the ground at rest, upright
    d_h = veh.hover_throttle()
    s0     = make_state(u=[d_h, d_h, 0, 0, 0], p=(0, 0, 50))
    s_goal = make_state(u=[d_h, d_h, 0, 0, 0], p=(0, 0, 0))

    sol = ocp.solve(s0, s_goal)
    S, Ud, info = ocp.extract_sol(sol)        # S: (N+1, 18) states, Ud: (N, 5) inputs u_dot
    print(f"      solver: {info['status']}, {info['iterations']} iterations, {info['solve_time']:.1f} s")

    # ------- check 1: the solver found a solution -------
    check("solver succeeded", info['success'], True, tol=0)

    # -------- check 2: symmetric landing -------
    check("delta_1 = delta_2 everywhere",   S[:, 13], S[:, 14], tol=1e-6)
    check("thrusters not used (m = 0)",     S[:, 15:18], 0,     tol=1e-6)
    check("no sideways motion (x, y = 0)",  S[:, 0:2], 0,       tol=1e-6)
    check("no rotation (w = 0)",            S[:, W], 0,         tol=1e-6)

    # ------- check 3: all bounds hold -------
    eps = 1e-6
    check("u >= lower bound",   np.all(S[:, U] >= veh.u_lb - eps), True, tol=0)
    check("u <= upper bound",   np.all(S[:, U] <= veh.u_ub + eps), True, tol=0)
    check("|u_dot| <= rate limit", np.all(np.abs(Ud) <= veh.u_dot_max + eps), True, tol=0)
    check("z >= 0 (not below ground)", np.all(S[:, 2] >= -eps), True, tol=0)

    # ------- check 4: it actually lands -------
    check("ends on the ground (z < 0.1 m)",     S[-1, 2], 0, tol=0.1)
    check("ends at rest (|v_z| < 0.1 m/s)",     S[-1, 5], 0, tol=0.1)

    # --- check 5: the OCP used the same model as the simulation ------------
    # re-simulate the optimal inputs with RK4 and compare with the OCP states
    f = lambda s, ud: np.array(veh.f(s, ud)).ravel()
    s = s0.copy()
    for k in range(ocp.N):
        s = rk4_step_ca(f, s, Ud[k], ocp.dt)
    check("re-simulated end state = OCP end state", s, S[-1], tol=1e-4)

    results[snap] = (veh, S)
    

    # ------- summary -------
print()
if n_failed == 0:
    print("all checks passed")
else:
    print(f"{n_failed} check(s) FAILED")



# ------- plot: both snapshots -------
t = np.arange(ocp.N + 1) * ocp.dt
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
for snap, (veh, S) in results.items():
    ax[0].plot(t, S[:, 2], label=snap)
    ax[1].plot(t, S[:, 5], label=snap)
    ax[2].plot(t, S[:, 13], label=f'{snap}: delta_1')
    ax[2].plot(t, S[:, 14], '--', label=f'{snap}: delta_2')
    ax[3].plot(t, S[:, 15:18])
ax[2].axhline(veh.u_lb[0], color='k', lw=0.8, ls=':')    # minimum throttle
ax[2].axhline(1.0, color='k', lw=0.8, ls=':')            # maximum throttle
ax[0].set_ylabel('z [m]')
ax[1].set_ylabel('v_z [m/s]')
ax[2].set_ylabel('throttle [-]')
ax[3].set_ylabel('m_x, m_y, m_z [-]')
ax[3].set_xlabel('t [s]')
ax[0].set_title('OCP: vertical landing from 50 m')
for a in ax:
    a.grid()
ax[0].legend()
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.show()