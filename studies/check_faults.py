import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.vehicle import Vehicle
from src.ocp import OCP_tracking
from src.faults import make_fault
from src.acs import acs_corners, hull_2d
from utils.integrator import rk4_step_ca

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# touchdown snapshot: light enough that one engine can still carry the lander
# (at burn start one engine (22 kN) < weight (25 kN) -> engine out is not recoverable)
config['vehicle']['snapshot'] = 'touchdown'
veh = Vehicle(config)
B = np.array(veh.B)

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

def tau_of(u, fault):
    # forces / moments of the casadi model for actuator values u (position etc. don't matter)
    s = veh.make_state(np.zeros(12), u=u)
    return np.array(veh.tau_fault(s, fault['lam'], fault['b_f'])).ravel()

nominal = make_fault(veh)
u_test = np.array([0.7, 0.4, 0.2, -0.3, 0.5])        # some actuator values inside U

print("\n=== part A: faults in the model ===")

# --- check 1: nominal fault = old model ----------------------------------
check("no fault: tau = B u", tau_of(u_test, nominal), B @ u_test, tol=1e-9)

# --- check 2: engine out removes exactly that column ---------------------
out1 = make_fault(veh, 'out', 0)
B_f = B * out1['lam']                                  # B Lambda_f (scales the columns)
check("engine 1 out: column 1 of B_f is zero",    B_f[:, 0], 0,        tol=1e-12)
check("engine 1 out: other columns unchanged",    B_f[:, 1:], B[:, 1:], tol=1e-12)
u_no1 = u_test.copy()
u_no1[0] = 0
check("engine 1 out: model ignores delta_1",      tau_of(u_test, out1), B @ u_no1, tol=1e-9)

# --- check 3: weak engine scales its contribution -------------------------
weak1 = make_fault(veh, 'weak', 0, 0.8)
part_weak = tau_of(u_test, weak1) - tau_of(u_no1, weak1)        # what engine 1 adds
part_nom = tau_of(u_test, nominal) - tau_of(u_no1, nominal)
check("engine 1 weak (0.8): contribution x 0.8",  part_weak, 0.8 * part_nom, tol=1e-9)

# --- check 4: stuck engine = out + constant shift --------------------------
stuck2 = make_fault(veh, 'stuck', 1, 0.3)
out2 = make_fault(veh, 'out', 1)
u_a, tau_stuck = acs_corners(veh, stuck2)
u_b, tau_out = acs_corners(veh, out2)
check("ACS has 2^5 = 32 corners",len(tau_stuck), 32,tol=0)
check("engine 2 stuck: ACS = ACS(out) + 0.3 b_2", tau_stuck, tau_out + 0.3 * B[:, 1], tol=1e-9)
u_other = u_test.copy()
u_other[1] = 0.9
check("engine 2 stuck: delta_2 command does nothing", tau_of(u_other, stuck2), tau_of(u_test, stuck2), tol=1e-9)

# --- check 5: slow engine only changes its rate limit ----------------------
slow1 = make_fault(veh, 'slow', 0, 0.3)
check("engine 1 slow (0.3): its rate limit x 0.3", slow1['u_dot_max'][0], 0.3 * veh.u_dot_max[0], tol=1e-12)
check("engine 1 slow (0.3): other limits unchanged", slow1['u_dot_max'][1:], veh.u_dot_max[1:], tol=1e-12)
check("engine 1 slow (0.3): B unchanged",         slow1['lam'], 1, tol=0)


# --- check 6: engine part of the ACS is a flat parallelogram --------------
u_e, tau_e = acs_corners(veh, nominal, which=(0, 1))     # 4 corners, thrusters at 0
P = tau_e[:, [1, 2, 3]]                                  # (F_y, F_z, M_x)
check("engines only: F_x, M_y, M_z are zero", tau_e[:, [0, 4, 5]], 0, tol=1e-9)
check("engines only: flat (rank 2 in F_y, F_z, M_x)", np.linalg.matrix_rank(P - P[0], tol=1e-6), 2, tol=0)
# corners in order (lo,lo), (lo,hi), (hi,lo), (hi,hi) -> parallelogram: P00 + P11 = P01 + P10
check("engines only: parallelogram",              P[0] + P[3], P[1] + P[2], tol=1e-9)


print("\n=== part B: the OCP knows the fault (touchdown, landing from 50 m) ===")

ocp = OCP_tracking(config, veh)
s0 = veh.make_state(config['simulation']['initial_state'])
s_goal = veh.make_state(config['simulation']['target_state'])
results = {}

for fault in [nominal, weak1, slow1, out1]:
    sol = ocp.solve(s0, s_goal, fault=fault)
    S, Ud, info = ocp.extract_sol(sol)
    tilt = np.degrees(2 * np.arccos(np.clip(np.abs(S[:, 6]), 0, 1)))   # angle away from upright
    print(f"\n  {fault['name']}: {info['status']}, {info['iterations']} iterations, "
          f"{info['solve_time']:.1f} s, max tilt {tilt.max():.1f} deg")

    check("solver succeeded",                     info['success'], True, tol=0)
    check("ends on the ground at rest",           S[-1, 0:6], s_goal[0:6], tol=1e-3)
    check("|u_dot| <= rate limits of this fault", np.all(np.abs(Ud) <= fault['u_dot_max'] + 1e-6), True, tol=0)

    # re-simulate the optimal inputs WITH the fault -> same end state as the OCP
    f = lambda s, ud: np.array(veh.f_fault(s, ud, fault['lam'], fault['b_f'])).ravel()
    s = s0.copy()
    for k in range(ocp.N):
        s = rk4_step_ca(f, s, Ud[k], ocp.dt)
    check("re-simulated with the fault = OCP end state", s, S[-1], tol=1e-4)

    results[fault['name']] = tilt

# engine out: the remaining engine pushes sideways -> the OCP must tilt the lander
check("engine 1 out: lander tilts (> 10 deg) to cancel the side force",
      results[out1['name']].max() > 10, True, tol=0)
check("no fault: lander stays upright", results[nominal['name']].max(), 0, tol=1e-6)



# --- summary -------------------------------------------------------------
print()
if n_failed == 0:
    print("all checks passed")
else:
    print(f"{n_failed} check(s) FAILED")



# --- plot: ACS projections -------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 6))

# left: engines only, (M_x, F_z) -> the parallelogram seen from the side
for fault, color in [(nominal, 'k'), (weak1, 'tab:orange'), (out1, 'tab:red'), (stuck2, 'tab:blue')]:
    _, tau = acs_corners(veh, fault, which=(0, 1))
    h = hull_2d(tau[:, [3, 2]] / 1e3)
    ax[0].plot(h[:, 0], h[:, 1], color=color, marker='o', label=fault['name'])
ax[0].set_xlabel('M_x [kNm]')
ax[0].set_ylabel('F_z [kN]')
ax[0].set_title('ACS of the engines (thrusters at 0), touchdown')

# right: whole ACS, moments M_x and M_y (engines + thrusters)
mx_out = make_fault(veh, 'out', 2)
for fault, color in [(nominal, 'k'), (mx_out, 'tab:red')]:
    _, tau = acs_corners(veh, fault)
    h = hull_2d(tau[:, [3, 4]] / 1e3)
    ax[1].plot(h[:, 0], h[:, 1], color=color, marker='o', label=fault['name'])
ax[1].set_xlabel('M_x [kNm]')
ax[1].set_ylabel('M_y [kNm]')
ax[1].set_title('ACS moments (all actuators), touchdown')

for a in ax:
    a.grid()
    a.legend()
plt.tight_layout()
plt.show()