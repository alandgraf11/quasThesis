import yaml
import numpy as np
from src.vehicle import Vehicle
from src.simulator import Simulator
from utils.plotting import plot_results
from utils.integrator import rk4_step_ca
from utils.quaternion import euler_to_q
from utils.animation import visualize_3d


if __name__ == '__main__':

    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # instantiate the vehicle dynamics
    vehicle = Vehicle(config=config)
    sim = Simulator(config=config, dynamics_func=vehicle.f, integrator=rk4_step_ca)

    # transform the initial state (euler angles in degree) to quaternions
    y0 = config['simulation']['initial_state']
    attitude = y0[6:9]  # attitude in euler angles in degree
    attitude = np.deg2rad(attitude) # attitude in euler angles in radians
    y0 = np.concatenate((y0[:6], attitude, y0[9:]))
    y0 = euler_to_q(euler=None, state=y0)

    # select the controls [Fx, Fy, Fz, taux, tauy, tauz] in body frame
    controls = [0,0,1*1.62,0,0,0] 

    # simulate the system
    time, states = sim.integrate(y0, controls, 0, config["simulation"]["duration"])

    if config['visualization']['plot']:
        plot_results(time, states, controls)

    if config['visualization']['animate']:
        visualize_3d(states, sim.dt)