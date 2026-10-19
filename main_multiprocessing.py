import csv
import time
import yaml
import shutil
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from src.vehicle import Vehicle
from src.ocp import OCP_tracking


_worker_ocp = None

def init_worker(config):
    """Initialize one independent OCP solver in each worker."""
    global _worker_ocp

    vehicle = Vehicle(config=config)
    vehicle.ca_model()

    ocp = OCP_tracking(config=config, dynamics=vehicle.f)
    ocp.init_solver()

    _worker_ocp = ocp

def solve_one_case(task):
    """Solve one OCP and return its results and timing information."""
    case_id, s0, s_goal = task
    start = time.perf_counter()

    try:
        sol = _worker_ocp.solve(s0, s_goal)
        S_opt, U_opt, info = _worker_ocp.extract_sol(sol)

        solve_time = time.perf_counter() - start

        # Extract a readable solver status from the info dictionary.
        if isinstance(info, dict):
            status = info.get(
                "return_status",
                info.get("status", "unknown"),
            )
            success = info.get("success", None)
        else:
            status = "unknown"
            success = None

        return {
            "case_id": case_id,
            "S_opt": np.asarray(S_opt),
            "U_opt": np.asarray(U_opt),
            "info": info,
            "solve_time": solve_time,
            "status": str(status),
            "success": success,
            "error": "",
        }

    except Exception as exc:
        return {
            "case_id": case_id,
            "S_opt": None,
            "U_opt": None,
            "info": None,
            "solve_time": time.perf_counter() - start,
            "status": "ERROR",
            "success": False,
            "error": repr(exc),
        }


if __name__ == "__main__":

    filename = time.strftime("%Y%m%d-%H%M%S") # name of the directory in results/

    results_dir = Path("results") / f"run_{filename}"
    results_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile("config.yaml", Path(results_dir) /"config.yaml")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_workers = config['batch']['num_cpus']
    N = config['batch']['N_cases']

    # sample random initial and target states
    rng = np.random.default_rng(42)

    start_lower_bounds = np.array([-2, -2, 0])
    start_upper_bounds = np.array([2, 2, 1])

    goal_lower_bounds = np.array([-3, -3, 0])
    goal_upper_bounds = np.array([3, 3, 5])

    start_conditions = rng.uniform(
        start_lower_bounds,
        start_upper_bounds,
        size=(N, 3),
    )

    goal_conditions = rng.uniform(
        goal_lower_bounds,
        goal_upper_bounds,
        size=(N, 3),
    )

    s0_array = np.zeros((N, 19))
    s_goal_array = np.zeros((N, 19))

    s0_array[:, 6] = 1.0
    s_goal_array[:, 6] = 1.0

    s0_array[:, 0:3] = start_conditions
    s_goal_array[:, 0:3] = goal_conditions

    # Package each independent problem into a task.
    tasks = [
        (i, s0_array[i], s_goal_array[i])
        for i in range(N)
    ]

    results = []

    log_path = results_dir / "log.csv"

    print(f"Solving {N} OCPs using {n_workers} worker processes.")

    start = time.perf_counter()

    # start multiprocessing and solve the OCPs
    with open(log_path, "w", newline="", encoding="utf-8") as log_file:

        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "case_id",
                "status",
                "success",
                "solve_time_s",
                "error",
            ],
        )

        writer.writeheader()
        log_file.flush()

        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(config,),
        ) as pool:

            for result in pool.imap_unordered(solve_one_case, tasks):

                case_id = result["case_id"]

                # Save the solution if the solve completed.
                if result["S_opt"] is not None:
                    np.savez_compressed(
                        results_dir / f"ocp_{case_id:03d}.npz",
                        s0=s0_array[case_id],
                        s_goal=s_goal_array[case_id],
                        S_opt=result["S_opt"],
                        U_opt=result["U_opt"],
                    )

                # Log the result in the main process only.
                writer.writerow({
                    "case_id": case_id,
                    "status": result["status"],
                    "success": result["success"],
                    "solve_time_s": f'{result["solve_time"]:.6f}',
                    "error": result["error"],
                })

                # Make the completed record visible on disk immediately.
                log_file.flush()

                print(
                    f'OCP {case_id + 1}/{N}: '
                    f'{result["status"]}, '
                    f'{result["solve_time"]:.2f} s'
                )

    end = time.perf_counter()

    print(f"Finished in {end - start:.2f} seconds.")
    print(f"Log saved to: {log_path.resolve()}")
    print(f"Results saved to: {results_dir.resolve()}")