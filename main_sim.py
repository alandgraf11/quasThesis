import yaml
import numpy as np
from src.vehicle import Vehicle
from src.simulator import Simulator
from utils.plotting import plot_results
from utils.integrator import rk4_step_ca
from utils.animation import visualize_3d


if __name__ == '__main__':

    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # instantiate the vehicle dynamics
    vehicle = Vehicle(config=config)
    sim = Simulator(config=config, dynamics_func=vehicle.f, integrator=rk4_step_ca)

    # initial state: 12 rigid-body values from config + actuators at hover throttle
    s0 = vehicle.make_state(config['simulation']['initial_state'])

    # constant input u_dot = rates of [delta_1, delta_2, m_x, m_y, m_z]
    u_dot = np.array(config['simulation']['u_dot'], dtype=float)

    # simulate the system (open loop)
    time, states = sim.integrate(s0, u_dot, 0, config["simulation"]["duration"])

    if config['visualization']['plot']:
        plot_results(time, states, u_dot, vehicle)

    if config['visualization']['animate']:
        visualize_3d(states, sim.dt)
