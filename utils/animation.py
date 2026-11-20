import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def quaternion_to_rotation_matrix(q):
    """
    Convert scalar-first quaternion [qw, qx, qy, qz]
    into a 3x3 body-to-world rotation matrix.
    """
    q = np.asarray(q, dtype=float).reshape(4)

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3)

    qw, qx, qy, qz = q / norm

    return np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx*qy - qw*qz),
            2 * (qx*qz + qw*qy),
        ],
        [
            2 * (qx*qy + qw*qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy*qz - qw*qx),
        ],
        [
            2 * (qx*qz - qw*qy),
            2 * (qy*qz + qw*qx),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])


def create_hopper_geometry():
    """
    Create a rocket-like vehicle in the body frame.

    Body +Z points toward the nose.
    """

    radius = 0.28
    body_length = 0.90
    nose_length = 0.42
    nozzle_length = 0.16
    nozzle_radius = 0.18
    fin_height = 0.28
    fin_length = 0.38
    n = 24

    vertices = []
    faces = []

    # --------------------------------------------------------------
    # Main cylindrical body
    # --------------------------------------------------------------

    rear_ring = []
    front_ring = []

    for i in range(n):
        theta = 2.0 * np.pi * i / n

        rear_ring.append(len(vertices))
        vertices.append([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ])

        front_ring.append(len(vertices))
        vertices.append([
            radius * np.cos(theta),
            radius * np.sin(theta),
            body_length,
        ])

    for i in range(n):
        j = (i + 1) % n
        faces.append([
            rear_ring[i],
            rear_ring[j],
            front_ring[j],
            front_ring[i],
        ])

    rear_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])

    for i in range(n):
        j = (i + 1) % n
        faces.append([
            rear_center,
            rear_ring[j],
            rear_ring[i],
        ])

    # --------------------------------------------------------------
    # Nose cone
    # --------------------------------------------------------------

    nose_tip = len(vertices)
    vertices.append([0.0, 0.0, body_length + nose_length])

    for i in range(n):
        j = (i + 1) % n
        faces.append([
            front_ring[i],
            front_ring[j],
            nose_tip,
        ])

    # --------------------------------------------------------------
    # Engine nozzle
    # --------------------------------------------------------------

    nozzle_outer_ring = []

    for i in range(n):
        theta = 2.0 * np.pi * i / n

        nozzle_outer_ring.append(len(vertices))
        vertices.append([
            nozzle_radius * np.cos(theta),
            nozzle_radius * np.sin(theta),
            -nozzle_length,
        ])

    for i in range(n):
        j = (i + 1) % n
        faces.append([
            rear_ring[i],
            nozzle_outer_ring[i],
            nozzle_outer_ring[j],
            rear_ring[j],
        ])

    nozzle_center = len(vertices)
    vertices.append([0.0, 0.0, -nozzle_length])

    for i in range(n):
        j = (i + 1) % n
        faces.append([
            nozzle_center,
            nozzle_outer_ring[i],
            nozzle_outer_ring[j],
        ])

    # --------------------------------------------------------------
    # Three fins
    # --------------------------------------------------------------

    fin_start_z = 0.08
    fin_end_z = fin_start_z + fin_length
    thickness = 0.035
    inner_radius = radius * 0.90
    outer_radius = radius + fin_height

    for k in range(3):
        theta = 2.0 * np.pi * k / 3.0

        radial = np.array([
            np.cos(theta),
            np.sin(theta),
            0.0,
        ])

        tangential = np.array([
            -np.sin(theta),
            np.cos(theta),
            0.0,
        ])

        p0 = radial * inner_radius + np.array(
            [0.0, 0.0, fin_start_z]
        ) + tangential * thickness

        p1 = radial * inner_radius + np.array(
            [0.0, 0.0, fin_end_z]
        ) + tangential * thickness

        p2 = radial * outer_radius + np.array(
            [0.0, 0.0, fin_start_z + 0.05]
        ) + tangential * thickness

        p3 = radial * outer_radius + np.array(
            [0.0, 0.0, fin_end_z - 0.08]
        ) + tangential * thickness

        p4 = radial * inner_radius + np.array(
            [0.0, 0.0, fin_start_z]
        ) - tangential * thickness

        p5 = radial * inner_radius + np.array(
            [0.0, 0.0, fin_end_z]
        ) - tangential * thickness

        p6 = radial * outer_radius + np.array(
            [0.0, 0.0, fin_start_z + 0.05]
        ) - tangential * thickness

        p7 = radial * outer_radius + np.array(
            [0.0, 0.0, fin_end_z - 0.08]
        ) - tangential * thickness

        idx = []

        for p in [p0, p1, p2, p3, p4, p5, p6, p7]:
            idx.append(len(vertices))
            vertices.append(p)

        faces.extend([
            [idx[0], idx[1], idx[3], idx[2]],
            [idx[4], idx[6], idx[7], idx[5]],
            [idx[1], idx[5], idx[7], idx[3]],
            [idx[0], idx[2], idx[6], idx[4]],
            [idx[2], idx[3], idx[7], idx[6]],
        ])

    return np.asarray(vertices), faces


def animate_hopper(
    states,
    dt,
    trail=True,
    show_body_axes=True,
    scale=4.0,
):
    """
    Animate a lander trajectory in real time (state layout: p, v, q, w, u).

    Parameters
    ----------
    states : ndarray
        Shape (N, n_states) or (N, n_states, 1); only p (0:3) and q (6:10) are used.

    dt : float
        Time between consecutive state samples, in seconds.

    trail : bool
        Show the trajectory.

    show_body_axes : bool
        Show body-fixed coordinate axes.

    scale : float
        Size factor for the drawn vehicle (1.0 -> about 1.5 m tall; 4.0 -> lander size).

    Returns
    -------
    fig : matplotlib Figure
    animation : FuncAnimation
    """

    states = np.asarray(states, dtype=float)

    # Normalize simulator/OCP state dimensions.
    if states.ndim == 3 and states.shape[-1] == 1:
        states = states[:, :, 0]

    if states.ndim != 2 or states.shape[1] < 13:
        raise ValueError(
            f"Expected states with shape (N, >=13) or (N, >=13, 1), "
            f"got {states.shape}"
        )

    if len(states) == 0:
        raise ValueError("States cannot be empty.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite, positive number.")

    if not np.all(np.isfinite(states[:, :13])):
        raise ValueError("States contain NaN or infinite values.")

    # State timestamps: t = 0, dt, 2*dt, ...
    t = dt * np.arange(len(states))
    positions = states[:, 0:3]

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------

    vertices_body, faces = create_hopper_geometry()
    vertices_body = vertices_body * scale

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Lander")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # Keep the camera limits fixed for the whole animation.
    rocket_size = 1.8 * scale

    mins = positions.min(axis=0) - rocket_size / 2
    maxs = positions.max(axis=0) + rocket_size / 2

    center = (mins + maxs) / 2
    span = max(np.max(maxs - mins), rocket_size) * 1.2

    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))

    # --------------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------------

    trajectory_line = None

    if trail:
        trajectory_line, = ax.plot(
            [], [], [],
            color="gray",
            alpha=0.6,
            linewidth=1.5,
            label="Trajectory",
        )

    # --------------------------------------------------------------
    # Rocket body
    # --------------------------------------------------------------

    body = Poly3DCollection(
        [],
        alpha=0.82,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.add_collection3d(body)

    # --------------------------------------------------------------
    # Body-fixed axes
    # --------------------------------------------------------------

    body_axes = []

    if show_body_axes:
        body_axes = [
            ax.plot([], [], [], color="red", linewidth=2)[0],
            ax.plot([], [], [], color="green", linewidth=2)[0],
            ax.plot([], [], [], color="blue", linewidth=3)[0],
        ]

    time_text = ax.text2D(
        0.05, 0.95, "",
        transform=ax.transAxes,
    )

    # --------------------------------------------------------------
    # Animation update
    # --------------------------------------------------------------

    def update(frame):
        state = states[frame]

        position = state[0:3]
        quat = state[6:10]

        # Body-frame to world-frame rotation.
        R = quaternion_to_rotation_matrix(quat)

        # Rotate and translate rocket vertices.
        vertices_world = vertices_body @ R.T + position

        body.set_verts([
            vertices_world[face]
            for face in faces
        ])

        # Update body coordinate axes.
        if show_body_axes:
            axis_length = 0.85 * scale

            for i, axis_line in enumerate(body_axes):
                endpoint = position + R[:, i] * axis_length

                axis_line.set_data(
                    [position[0], endpoint[0]],
                    [position[1], endpoint[1]],
                )

                axis_line.set_3d_properties(
                    [position[2], endpoint[2]]
                )

        # Update trajectory trail.
        if trail:
            trajectory = positions[:frame + 1]

            trajectory_line.set_data(
                trajectory[:, 0],
                trajectory[:, 1],
            )

            trajectory_line.set_3d_properties(
                trajectory[:, 2]
            )

        time_text.set_text(f"t = {t[frame]:.3f} s")

        artists = [body, time_text]

        if show_body_axes:
            artists.extend(body_axes)

        if trail:
            artists.append(trajectory_line)

        return artists

    # --------------------------------------------------------------
    # Real-time animation
    # --------------------------------------------------------------

    # One state advances every dt seconds.
    # Matplotlib expects milliseconds.
    interval_ms = dt * 1000.0

    animation = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=interval_ms,
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )

    return fig, animation


def visualize_3d(states, dt, trail=True, show_body_axes=True, scale=4.0):
    """
    Display the rocket animation in real time.

    Parameters
    ----------
    states : ndarray
        Shape (N, n_states) or (N, n_states, 1).

    dt : float
        Time step between consecutive states, in seconds.

    scale : float
        Size factor for the drawn vehicle.
    """

    fig, animation = animate_hopper(
        states=states,
        dt=dt,
        trail=trail,
        show_body_axes=show_body_axes,
        scale=scale,
    )

    plt.show()

    return fig, animation