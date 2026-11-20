import itertools
import numpy as np
from scipy.spatial import ConvexHull, QhullError

def acs_corners(vehicle, fault, which=(0,1,2,3,4)):
    # corners of the ACS: every corner of the box U, mapped through tau = B (lam * u) + b_f
    #   which: indices of the actuators that move between their bounds,
    #          all others are set to 0 (= they add nothing), e.g. which=(0, 1) -> engines only
    # returns u (K x 5) and tau (K x 6), K = 2^len(which)
    B = np.array(vehicle.B)
    u_list = []
    for combo in itertools.product([0, 1], repeat=len(which)): # all lower/upper combinations
        u = np.zeros(vehicle.n_u)
        for k, i in enumerate(which):
            u[i] = vehicle.u_ub[i] if combo[k] == 1 else vehicle.u_lb[i]
        u_list.append(u)
    u = np.array(u_list)
    tau = (u * fault['lam']) @ B.T + fault['b_f']                  # one row per corner
    return u, tau

def hull_2d(points):
    # outline of 2D points for plotting: hull corners in counterclockwise order, closed
    points = np.asarray(points)
    try:
        corners = points[ConvexHull(points).vertices]       # vertices = indices of the corners, in order
    except QhullError:
        # flat set (all points on one line, e.g. one engine out): no area -> the points sorted along the line
        corners = np.unique(points.round(9), axis=0)
    return np.vstack([corners, corners[:1]])                # repeat first point to close the polygon