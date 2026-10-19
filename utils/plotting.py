import numpy as np
import matplotlib.pyplot as plt
from utils.quaternion import q_to_euler

def plot_results(time, states, controls, s_goal=None, title=None):
    """
    Plot OCP or simulation results using a common interface.

    OCP:
        time     : (N+1,)
        states   : (N+1, 19)
        controls : (N, 6)

    Simulation:
        time     : (N,)
        states   : (N, 19, 1) or (N, 19)
        controls : (6,)

    s_goal:
        Optional reference state, shape (19,) or (N_states, 19).
    """

    time = np.asarray(time, dtype=float).ravel()
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)

    # ------------------------------------------------------------
    # Normalize states
    # ------------------------------------------------------------

    if states.ndim == 3 and states.shape[-1] == 1:
        states = states.squeeze(axis=-1)

    if states.ndim != 2 or states.shape[1] < 19:
        raise ValueError(
            f"Expected states with shape (N, >=19), got {states.shape}"
        )

    if len(time) != len(states):
        raise ValueError(
            f"Time length {len(time)} != number of state samples "
            f"{len(states)}"
        )

    # ------------------------------------------------------------
    # Identify mode and prepare control history
    # ------------------------------------------------------------

    is_ocp = controls.ndim == 2

    if is_ocp:
        if controls.shape[1] != 6:
            raise ValueError(
                f"Expected OCP controls with shape (N, 6), "
                f"got {controls.shape}"
            )

        if controls.shape[0] != len(states) - 1:
            raise ValueError(
                "OCP controls must have one row fewer than states"
            )

        if len(time) != controls.shape[0] + 1:
            raise ValueError(
                "OCP time must have one sample more than controls"
            )

        if controls.shape[0] == 0:
            raise ValueError("OCP controls must contain at least one row")

        # Repeat the final control to align with the state time array.
        U_plot = np.vstack((controls, controls[-1]))

        if title is None:
            title = "OCP Solution"

    else:
        if controls.shape != (6,):
            raise ValueError(
                f"Expected simulation controls with shape (6,), "
                f"got {controls.shape}"
            )

        # Constant controls throughout the simulation.
        U_plot = np.tile(controls, (len(time), 1))

        if title is None:
            title = "Vehicle Simulation Results"

    # ------------------------------------------------------------
    # Extract states
    # ------------------------------------------------------------

    def unpack(S):
        return {
            "pos": S[:, 0:3],
            "vel": S[:, 3:6],
            "euler": np.rad2deg(q_to_euler(S[:, 6:10])),
            "omega": np.rad2deg(S[:, 10:13]),
            "F": S[:, 13:16],
            "tau": S[:, 16:19],
        }

    d = unpack(states)

    # ------------------------------------------------------------
    # Optional reference trajectory
    # ------------------------------------------------------------

    g = None

    if s_goal is not None:
        s_goal = np.asarray(s_goal, dtype=float)

        if s_goal.ndim == 1:
            if s_goal.size < 19:
                raise ValueError("s_goal must contain at least 19 states")
            s_goal = np.tile(s_goal, (len(time), 1))

        if s_goal.ndim != 2 or s_goal.shape[1] < 19:
            raise ValueError(
                f"Expected s_goal with shape (19,) or (N, >=19), "
                f"got {s_goal.shape}"
            )

        if len(s_goal) != len(time):
            raise ValueError(
                "s_goal must have one row per state time sample"
            )

        g = unpack(s_goal)

    # ------------------------------------------------------------
    # Plot configuration
    # ------------------------------------------------------------

    axis_names = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    angle_names = ["Roll (phi)", "Pitch (theta)", "Yaw (psi)"]

    fig, axs = plt.subplots(
        2, 4,
        figsize=(21, 9),
        sharex=True,
    )

    fig.suptitle(title, fontsize=16)

    def format_axis(ax, plot_title, ylabel):
        ax.set_title(plot_title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    def plot_state(ax, key, labels, plot_title, ylabel):
        for i in range(3):
            ax.plot(
                time, d[key][:, i],
                color=colors[i],
                label=labels[i],
            )

            if g is not None:
                ax.plot(
                    time, g[key][:, i],
                    color=colors[i],
                    linestyle="--",
                    alpha=0.6,
                    label=f"{labels[i]} (goal)",
                )

        format_axis(ax, plot_title, ylabel)

    def plot_control(ax, indices, labels, plot_title, ylabel):
        for i, idx in enumerate(indices):
            ax.step(
                time, U_plot[:, idx],
                where="post",
                color=colors[i],
                label=labels[i],
            )

        format_axis(ax, plot_title, ylabel)

    # ------------------------------------------------------------
    # Row 1
    # ------------------------------------------------------------

    plot_state(
        axs[0, 0], "pos", axis_names,
        "Position", "Position [m]",
    )

    plot_state(
        axs[0, 1], "vel", axis_names,
        "Velocity", "Velocity [m/s]",
    )

    plot_state(
        axs[0, 2], "F", [f"F{a}" for a in axis_names],
        "Forces", "Force [N]",
    )

    plot_control(
        axs[0, 3], [0, 1, 2],
        [f"dF{a}/dt" for a in axis_names],
        "Force Rates (Controls)", "Force rate [N/s]",
    )

    # ------------------------------------------------------------
    # Row 2
    # ------------------------------------------------------------

    plot_state(
        axs[1, 0], "euler", angle_names,
        "Euler Angles", "Angle [deg]",
    )

    plot_state(
        axs[1, 1], "omega", [f"{a} rate" for a in axis_names],
        "Angular Rates", "Angular rate [deg/s]",
    )

    plot_state(
        axs[1, 2], "tau", [f"Tau {a}" for a in axis_names],
        "Torques", "Torque [N m]",
    )

    plot_control(
        axs[1, 3], [3, 4, 5],
        [f"dTau{a}/dt" for a in axis_names],
        "Torque Rates (Controls)", "Torque rate [N m/s]",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
