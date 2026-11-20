import numpy as np
import matplotlib.pyplot as plt
from utils.quaternion import q_to_euler


def plot_results(time, states, controls, vehicle, s_goal=None, title=None, fault=None):
    """
    Plot OCP or simulation results (state layout: p, v, q, w, u -> 18 states).

    OCP:
        time     : (N+1,)
        states   : (N+1, 18)
        controls : (N, 5)      u_dot at every OCP step

    Simulation:
        time     : (N,)
        states   : (N, 18, 1) or (N, 18)
        controls : (5,)        constant u_dot

    vehicle:
        Vehicle object; gives B (forces / moments), the actuator bounds and the rate limits.

    s_goal:
        Optional goal state, shape (18,) or (len(time), 18); drawn dashed for p, v, euler, w.
    """

    time = np.asarray(time, dtype=float).ravel()
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)

    if states.ndim == 3 and states.shape[-1] == 1:
        states = states.squeeze(axis=-1)

    n_u = vehicle.n_u
    if states.shape != (len(time), vehicle.n_states):
        raise ValueError(f"Expected states with shape ({len(time)}, {vehicle.n_states}), "
                         f"got {states.shape}")

    # ------------------------------------------------------------
    # inputs u_dot: one row per state sample (for step plots)
    # ------------------------------------------------------------

    if controls.ndim == 2:                          # OCP: one u_dot per step
        if controls.shape != (len(time) - 1, n_u):
            raise ValueError(f"Expected OCP controls with shape ({len(time) - 1}, {n_u}), "
                             f"got {controls.shape}")
        U_dot = np.vstack((controls, controls[-1]))     # repeat last one to match the time axis
        if title is None:
            title = "OCP Solution"
    else:                                           # simulation: constant u_dot
        if controls.shape != (n_u,):
            raise ValueError(f"Expected simulation controls with shape ({n_u},), "
                             f"got {controls.shape}")
        U_dot = np.tile(controls, (len(time), 1))
        if title is None:
            title = "Vehicle Simulation Results"

    # ------------------------------------------------------------
    # split the states
    # ------------------------------------------------------------

    def unpack(S):
        return {
            "pos": S[:, 0:3],
            "vel": S[:, 3:6],
            "euler": np.rad2deg(q_to_euler(S[:, 6:10])),
            "omega": np.rad2deg(S[:, 10:13]),
        }

    d = unpack(states)
    u = states[:, 13:13 + n_u]                      # actuators: delta_1, delta_2, m_x, m_y, m_z

    # forces and moments from the actuators: tau = B (lam u) + b_f (body frame)
    B = np.array(vehicle.B)
    if fault is None:
        tau = u @ B.T                               # (N, 6), no fault
    else:
        tau = (u * fault['lam']) @ B.T + fault['b_f']
    F_body = tau[:, 0:3] / 1e3                      # kN
    M_body = tau[:, 3:6] / 1e3                      # kNm

    # what the actuators really produce (fault included); u itself is only the command
    u_eff = u.copy()
    if fault is not None:
        u_eff = u * fault['lam']                    # weak: scaled, out: 0
        if fault['kind'] == 'stuck':
            u_eff[:, fault['actuator']] = fault['value']   # stuck: constant, whatever the command

    g = None
    if s_goal is not None:
        s_goal = np.asarray(s_goal, dtype=float)
        if s_goal.ndim == 1:
            s_goal = np.tile(s_goal, (len(time), 1))
        g = unpack(s_goal)

    # ------------------------------------------------------------
    # figure
    # ------------------------------------------------------------

    axis_names = ["x", "y", "z"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    angle_names = ["roll (phi)", "pitch (theta)", "yaw (psi)"]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    fig.suptitle(title, fontsize=16)

    def format_axis(ax, plot_title, ylabel):
        ax.set_title(plot_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    def plot_three(ax, data, labels, plot_title, ylabel, goal=None):
        for i in range(3):
            ax.plot(time, data[:, i], color=colors[i], label=labels[i])
            if goal is not None:
                ax.plot(time, goal[:, i], color=colors[i], linestyle="--", alpha=0.6,
                        label=f"{labels[i]} (goal)")
        format_axis(ax, plot_title, ylabel)

    def limits(ax, low, high):
        # dotted black lines at the allowed range
        ax.axhline(low, color="k", linestyle=":", linewidth=1)
        ax.axhline(high, color="k", linestyle=":", linewidth=1)

    # --- row 1: translation and attitude -----------------------------------
    plot_three(axs[0, 0], d["pos"], axis_names, "Position (world)", "Position [m]",
               g["pos"] if g else None)
    plot_three(axs[0, 1], d["vel"], [f"v_{a}" for a in axis_names], "Velocity (world)",
               "Velocity [m/s]", g["vel"] if g else None)
    plot_three(axs[0, 2], d["euler"], angle_names, "Euler angles", "Angle [deg]",
               g["euler"] if g else None)

    # --- row 2: rotation and actuators -------------------------------------
    plot_three(axs[1, 0], d["omega"], [f"w_{a}" for a in axis_names], "Rotation rates (body)",
               "Rate [deg/s]", g["omega"] if g else None)

    # solid = what the actuator produces, thin dashed = command (only drawn if it differs)
    ax = axs[1, 1]
    for i, (name, color) in enumerate([("delta_1 (engine 1)", "tab:orange"),
                                       ("delta_2 (engine 2)", "tab:purple")]):
        ax.plot(time, u_eff[:, i], color=color, label=name)
        if not np.allclose(u_eff[:, i], u[:, i]):
            ax.plot(time, u[:, i], color=color, linestyle="--", linewidth=1, alpha=0.7,
                    label=f"{name.split()[0]} command")
    limits(ax, vehicle.u_lb[0], vehicle.u_ub[0])
    format_axis(ax, "Engine throttle (produced)", "Throttle [-]")

    ax = axs[1, 2]
    for i in range(3):
        ax.plot(time, u_eff[:, 2 + i], color=colors[i], label=f"m_{axis_names[i]}")
        if not np.allclose(u_eff[:, 2 + i], u[:, 2 + i]):
            ax.plot(time, u[:, 2 + i], color=colors[i], linestyle="--", linewidth=1, alpha=0.7,
                    label=f"m_{axis_names[i]} command")
    limits(ax, vehicle.u_lb[2], vehicle.u_ub[2])
    format_axis(ax, "RCS moments (produced)", "Fraction of max [-]")

    # --- row 3: forces, moments, inputs -----------------------------------
    plot_three(axs[2, 0], F_body, [f"F_{a}" for a in axis_names],
               "Forces from actuators (body)", "Force [kN]")
    plot_three(axs[2, 1], M_body, [f"M_{a}" for a in axis_names],
               "Moments from actuators (body)", "Moment [kNm]")

    ax = axs[2, 2]
    input_names = ["delta_1", "delta_2", "m_x", "m_y", "m_z"]
    input_colors = ["tab:orange", "tab:purple", "tab:red", "tab:green", "tab:blue"]
    for i in range(n_u):
        ax.step(time, U_dot[:, i], where="post", color=input_colors[i],
                label=f"d/dt {input_names[i]}")
    limits(ax, -vehicle.u_dot_max[0], vehicle.u_dot_max[0])   # engine rate limit
    format_axis(ax, "Inputs u_dot (dotted: engine rate limit)", "Rate [1/s]")

    for ax in axs[2, :]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
