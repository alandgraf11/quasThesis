import csv
import time
from collections import Counter
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

def solution_quality(S_opt, U_opt, s_goal, ocp):
    """Compute quality metrics of one solved trajectory."""
    s_N = S_opt[-1]

    # terminal errors w.r.t. the goal state
    pos_err = np.linalg.norm(s_N[0:3] - s_goal[0:3])
    vel_err = np.linalg.norm(s_N[3:6] - s_goal[3:6])
    angvel_err = np.linalg.norm(s_N[10:13] - s_goal[10:13])

    # attitude error as rotation angle between the two quaternions
    q = s_N[6:10] / np.linalg.norm(s_N[6:10])
    q_goal = s_goal[6:10] / np.linalg.norm(s_goal[6:10])
    att_err_deg = np.degrees(2 * np.arccos(np.clip(abs(q @ q_goal), 0.0, 1.0)))

    # largest deviation of the quaternion norm from 1 along the trajectory
    quat_norm_dev = np.max(np.abs(np.linalg.norm(S_opt[:, 6:10], axis=1) - 1))

    # largest violation of the box constraints on states and controls
    x_opt = np.concatenate([S_opt.ravel(), U_opt.ravel()])
    bound_viol = max(
        np.max(np.maximum(ocp.lbx - x_opt, 0.0)),
        np.max(np.maximum(x_opt - ocp.ubx, 0.0)),
    )

    # control effort: integral of the squared control rates
    control_effort = np.sum(U_opt ** 2) * ocp.dt

    return {
        "final_pos_err": pos_err,
        "final_vel_err": vel_err,
        "final_att_err_deg": att_err_deg,
        "final_angvel_err": angvel_err,
        "quat_norm_dev": quat_norm_dev,
        "bound_viol": bound_viol,
        "control_effort": control_effort,
    }

def solve_one_case(task):
    """Solve one OCP and return its results and timing information."""
    case_id, s0, s_goal = task
    start = time.perf_counter()

    try:
        sol = _worker_ocp.solve(s0, s_goal)
        S_opt, U_opt, info = _worker_ocp.extract_sol(sol)

        solve_time = time.perf_counter() - start

        S_opt = np.asarray(S_opt)
        U_opt = np.asarray(U_opt)

        return {
            "case_id": case_id,
            "S_opt": S_opt,
            "U_opt": U_opt,
            "info": info,
            "solve_time": solve_time,
            "status": str(info["status"]),
            "success": info["success"],
            "iterations": info["iterations"],
            "cost": info["cost"],
            "quality": solution_quality(S_opt, U_opt, s_goal, _worker_ocp),
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
            "iterations": None,
            "cost": None,
            "quality": {},
            "error": repr(exc),
        }


QUALITY_FIELDS = [
    "final_pos_err",
    "final_vel_err",
    "final_att_err_deg",
    "final_angvel_err",
    "quat_norm_dev",
    "bound_viol",
    "control_effort",
]

def summarize(rows, total_time):
    """Build a text summary of the whole batch from the logged rows."""
    n = len(rows)
    n_success = sum(1 for r in rows if r["success"] is True)
    lines = [
        f"Cases:            {n}",
        f"Successful:       {n_success}/{n} ({100 * n_success / n:.1f} %)",
        f"Total wall time:  {total_time:.2f} s",
        "",
        "Solver status counts:",
    ]
    status_counts = Counter(r["status"] for r in rows)
    for status, count in status_counts.most_common():
        lines.append(f"  {status:<35} {count}")

    errors = [r for r in rows if r["error"]]
    if errors:
        lines.append("")
        lines.append("Cases with exceptions:")
        for r in errors:
            lines.append(f"  {r['case_id']}: {r['error']}")

    # statistics of the numeric columns, separately for all solved cases
    # and for the successful ones only
    def stats_table(subset, title):
        lines.append("")
        lines.append(f"{title} (n={len(subset)})")
        lines.append(f"  {'metric':<20}{'mean':>12}{'median':>12}{'min':>12}{'max':>12}")
        for key in ["solve_time_s", "iterations", "cost"] + QUALITY_FIELDS:
            values = np.array([float(r[key]) for r in subset if r[key] != ""])
            if values.size == 0:
                continue
            lines.append(
                f"  {key:<20}{np.mean(values):>12.4g}{np.median(values):>12.4g}"
                f"{np.min(values):>12.4g}{np.max(values):>12.4g}"
            )

    solved = [r for r in rows if not r["error"]]
    successful = [r for r in solved if r["success"] is True]
    stats_table(solved, "All solved cases")
    if len(successful) != len(solved):
        stats_table(successful, "Successful cases")

    # the worst cases are the ones worth looking at first
    worst = sorted(solved, key=lambda r: float(r["final_pos_err"]), reverse=True)[:5]
    lines.append("")
    lines.append("Largest final position errors:")
    for r in worst:
        lines.append(
            f"  case {r['case_id']:>4}: {float(r['final_pos_err']):.4g} m ({r['status']})"
        )

    return "\n".join(lines)


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

    rows = []

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
                "iterations",
                "cost",
                *QUALITY_FIELDS,
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
                row = {
                    "case_id": case_id,
                    "status": result["status"],
                    "success": result["success"],
                    "solve_time_s": f'{result["solve_time"]:.6f}',
                    "iterations": result["iterations"] if result["iterations"] is not None else "",
                    "cost": f'{result["cost"]:.6g}' if result["cost"] is not None else "",
                    "error": result["error"],
                }
                for key in QUALITY_FIELDS:
                    value = result["quality"].get(key)
                    row[key] = f"{value:.6g}" if value is not None else ""
                writer.writerow(row)
                rows.append(row)

                # Make the completed record visible on disk immediately.
                log_file.flush()

                print(
                    f'OCP {case_id + 1}/{N}: '
                    f'{result["status"]}, '
                    f'{result["solve_time"]:.2f} s'
                )

    end = time.perf_counter()

    summary = summarize(rows, end - start)
    summary_path = results_dir / "1summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print(f"Finished in {end - start:.2f} seconds.")
    print(f"Log saved to: {log_path.resolve()}")
    print(f"Summary saved to: {summary_path.resolve()}")
    print(f"Results saved to: {results_dir.resolve()}")