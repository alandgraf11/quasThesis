# QUAS

QUAS (QUantitative ASsessment) is the common starting point for your BTh. It provides a point mass model (rigid body) of a generic spacecraft (no atmosphere, quaternion based attitude).

## Features

- 6-degree-of-freedom translational and rotational vehicle dynamics
- Scalar-first quaternion attitude representation
- 19 dimensional state vector -> 13 vehicle states and 6 force / torque states (integrator dynamics)
- 6 dimensional control input -> force rates and torque rates
- Numerical simulation of the Vehicle
- Finite horizon optimal control problem (NLP) discretized with RK4, formulated in CasADi solved via IPOPT
- State and Control trajectory plotting as well as 3D animation
- Multiprocessing based batch solving of OCP including results logging

## Not inluded / Assumptions
- Mass and inertia are constant instead of time varying
- Only principal axis inertia is unequal zero, i.e. Ixy=Ixz=Iyz=0
- Isp is indepdenent of operationg point
- No atmosphere
- Economic OCP formulation (propellant efficient trajectory for instance)

## Requirements

- Python
- Packages listed in [`requirements.txt`](requirements.txt): NumPy, CasADi, Matplotlib, and PyYAML

## Usage

Run commands from the repository root so scripts can find `config.yaml`.

```bash
# Open config.yaml and make the changes you desire -> almost everything is controlled from the config file

# Simulate the vehicle with the configured initial state and fixed controls and plot / animate it -> controls need to be chosen in main_sim.py
python main_sim.py

# Solve one optimal-control tracking problem and plot / animate it
python main_ocp.py

# Solve a batch of independently generated OCP cases in parallel and log them in /results/
python main_multiprocessing.py
```

The simulation and OCP entry points use the `visualization` settings in `config.yaml` to enable plots and/or the 3D animation. The batch runner writes its output to a timestamped subdirectory under `results/`.

## Configuration

Edit [`config.yaml`](config.yaml) to configure:

- `vehicle`: mass and moments of inertia
- `simulation`: integration step, duration, and initial state
- `visualization`: whether to plot results or animate the vehicle
- `ocp`: prediction horizon, time step, cost weights, and force, torque, and rate bounds
- `batch`: number of worker processes and number of cases

The dynamics state has 19 values, in this order:

```text
[x, y, z, u, v, w, qw, qx, qy, qz, p, q, r, Fx, Fy, Fz, taux, tauy, tauz]
```

Position is in meters, velocity in meters per second, angular rates in radians per second, forces in newtons, and torques in newton-meters. Attitude is stored internally as a scalar-first quaternion `[qw, qx, qy, qz]`. `simulation.initial_state` is supplied with three Euler angles (from `config.yaml`) in degrees at indices 6-8; the `main_XXX.py` scripts convert those angles to a quaternion before solving or integrating. The angles for initial and terminal state are provided as euler angles in `config.yaml` as euler angles are more intuitive to think about than quaternions. This can be adapted at any time off course. 

The six controls are force and torque rates:

```text
[dFx/dt, dFy/dt, dFz/dt, dtaux/dt, dtauy/dt, dtauz/dt]
```

The vehicle model defaults to lunar gravity (`g = 1.62 m/s^2`).

## Batch Output -> `main_multiprocessing.py`

Each run creates `results/run_<timestamp>/` containing:

- `config.yaml`: a copy of the configuration used for the run
- `log.csv`: case ID, solver status, success flag, solve time, and any error
- `ocp_<case>.npz`: compressed start/goal states and solved state/control trajectories for successful cases

## Project Structure

```text
config.yaml                 Runtime and solver configuration
main_sim.py                 Single vehicle simulation entry point
main_ocp.py                 Single OCP tracking entry point
main_multiprocessing.py     Parallel batch OCP entry point
src/
	vehicle.py                Vehicle dynamics model
	simulator.py              Time-stepping simulation wrapper
	ocp.py                    OCP tracking problem and solver
utils/
	integrator.py             Runge-Kutta 4 integrator
	quaternion.py             Euler/quaternion conversion helpers
	animation.py              3D vehicle animation; vibe-coded
	plotting.py               Shared simulation/OCP result plots; vibe-coded
results/                    Batch output directory
```

## Notes

- The example batch runner generates random start and goal positions. This only serves as an example for how to call the OCP in a multiprocessing batch process. 
- Initial states must not violate the state constraints, both defined in the config.yaml -> this can make the solver fail to converge

## Notes for Felix

- Linearization
- Casadi Graphs
- Actuator Set
- Solve log html stream