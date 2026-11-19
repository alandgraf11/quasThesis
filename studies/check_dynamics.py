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

def simulate(vehicle, s0, u_dot, t_end, dt=0.1, normalize=True):
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