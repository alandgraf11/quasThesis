import numpy as np
import casadi as ca
import yaml
from src.actuators import build_B

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
cfg = config['propulsion']
L_V = [snap['l_v'] for snap in config['vehicle']['snapshots'].values()]   # [0.81, 1.57]

n_failed = 0



def check(name, value, expected, tol=1.0):
    # compares value with expected (tolerance tol) and prints PASS or FAIL
    global n_failed
    ok = np.allclose(value, expected, atol=tol)
    if ok:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}: got {value}, expected {expected}")
        n_failed += 1



# --- check 1: engine columns against Felix's formula -------------------
T = cfg['T_max']
beta = np.deg2rad(cfg['beta_deg'])
l_l = cfg['l_l']

for l_v in L_V:
    B = np.array(build_B(cfg, l_v))

    F_y = T * np.cos(beta)
    F_z = T * np.sin(beta)
    M_x = T * (np.sin(beta) * l_l - np.cos(beta) * l_v)   # Felix's formula

    check(f"engine 1 F_y  (l_v = {l_v})", B[1, 0], -F_y)
    check(f"engine 1 F_z  (l_v = {l_v})", B[2, 0], F_z)
    check(f"engine 1 M_x  (l_v = {l_v})", B[3, 0], M_x)
    check(f"engine 2 F_y  (l_v = {l_v})", B[1, 1], F_y)
    check(f"engine 2 F_z  (l_v = {l_v})", B[2, 1], F_z)
    check(f"engine 2 M_x  (l_v = {l_v})", B[3, 1], -M_x)



# --- check 2: F_x row, RCS diagonal, rank -------------------------------
B = np.array(build_B(cfg, L_V[0]))
rcs = cfg['rcs']

check("F_x row is zero", B[0, :], np.zeros(5))
for j, axis in enumerate(['x', 'y', 'z']):
    M_rcs = rcs['n_per_axis'] * rcs['F'] * rcs['lever'][j]
    check(f"RCS M_{axis}", B[3 + j, 2 + j], M_rcs)
check("rank of B is 5", np.linalg.matrix_rank(B), 5, tol=0)




# --- check 3: thrust line through the CoG -> no engine moment -----------
for l_v in L_V:
    cfg_a0 = dict(cfg)                                   # copy of the parameters
    cfg_a0['beta_deg'] = np.rad2deg(np.arctan(l_v / l_l))
    B_a0 = np.array(build_B(cfg_a0, l_v))
    print(f"      (beta for a = 0 at l_v = {l_v}: {cfg_a0['beta_deg']:.2f} deg)")
    check(f"engine M_x = 0 at that beta", B_a0[3, 0:2], np.zeros(2))




# --- check 4: B also works with a symbolic l_v (needed in stage 3) ------
l_v_sym = ca.SX.sym('l_v')
B_sym = build_B(cfg, l_v_sym)
B_func = ca.Function('B_func', [l_v_sym], [B_sym])
for l_v in L_V:
    check(f"symbolic B = numeric B  (l_v = {l_v})",
          np.array(B_func(l_v)), np.array(build_B(cfg, l_v)))




# --- summary -------------------------------------------------------------
print()
if n_failed == 0:
    print("all checks passed")
else:
    print(f"{n_failed} check(s) FAILED")