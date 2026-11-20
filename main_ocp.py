import yaml
import numpy as np
from src.vehicle import Vehicle
from src.ocp import OCP_tracking
from src.faults import make_fault
from utils.animation import visualize_3d
from utils.plotting import plot_results



if __name__ == '__main__':

    # Load configuration from YAML file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # instantiate the vehicle dynamics
    vehicle = Vehicle(config=config)

    # start and goal: 12 rigid-body values from config + actuators at hover throttle
    s0 = vehicle.make_state(config['simulation']['initial_state'])
    s_goal = vehicle.make_state(config['simulation']['target_state'])

    # fault from config.yaml (kind: none -> nominal)
    fault = make_fault(vehicle, **config['fault'])
    print(f"fault: {fault['name']}")

    # instantiate the OCP solver
    ocp = OCP_tracking(config=config, vehicle=vehicle)
    # solve the OCP
    sol = ocp.solve(s0, s_goal, fault=fault)
    # extract the solution
    S_opt, U_opt, info = ocp.extract_sol(sol)
    print(f"solver: {info['status']}, {info['iterations']} iterations, "
          f"{info['solve_time']:.1f} s, cost {info['cost']:.4g}")

    dt = ocp.dt
    N = ocp.N
    time = np.linspace(0, dt*N, N+1)

    if config['visualization']['plot']:
        plot_results(time, S_opt, U_opt, vehicle, s_goal=s_goal, fault=fault,
                     title=f"OCP Solution ({fault['name']})")

    if config['visualization']['animate']:
        visualize_3d(S_opt, dt)