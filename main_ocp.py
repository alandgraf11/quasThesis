import yaml
import numpy as np
from src.vehicle import Vehicle
from src.simulator import Simulator
from src.ocp import OCP_tracking
from utils.integrator import rk4_step_ca
from utils.quaternion import euler_to_q
from utils.animation import visualize_3d
from utils.plotting import plot_results



if __name__ == '__main__':
    
    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # instantiate the vehicle dynamics
    vehicle = Vehicle(config=config)
    sim = Simulator(config=config, dynamics_func=vehicle.f, integrator=rk4_step_ca)

    # transform the initial and terminal state (euler angles in degree) to quaternions
    y0 = config['simulation']['initial_state']
    attitude = y0[6:9]  # attitude in euler angles in degree
    attitude = np.deg2rad(attitude) # attitude in euler angles in radians
    y0 = np.concatenate((y0[:6], attitude, y0[9:]))
    y0 = euler_to_q(euler=None, state=y0)
    s0 = y0

    s_goal = config['simulation']['target_state']
    attitude = s_goal[6:9]  # attitude in euler angles in degree
    attitude = np.deg2rad(attitude) # attitude in euler angles in radians
    s_goal = np.concatenate((s_goal[:6], attitude, s_goal[9:]))
    s_goal = euler_to_q(euler=None, state=s_goal)

    # instantiate the OCP solver
    ocp = OCP_tracking(config=config, dynamics=vehicle.f)
    # solve the OCP
    sol = ocp.solve(s0, s_goal)
    # extract the solution
    S_opt, U_opt, info = ocp.extract_sol(sol)

    dt = ocp.dt
    N = ocp.N
    time = np.linspace(0, dt*N, N+1)

    if config['visualization']['plot']:
        plot_results(time, S_opt, U_opt)

    if config['visualization']['animate']:
        visualize_3d(S_opt, dt)