import sys
import numpy as np


# state and control conventions of the OCP (see src/vehicle.py)
STATE_NAMES = [
    "x", "y", "z", "vx", "vy", "vz",
    "qw", "qx", "qy", "qz", "wx", "wy", "wz",
    "d1", "d2", "mx", "my", "mz",            # actuators: throttles, RCS moment commands
]
CONTROL_NAMES = ["d1_dot", "d2_dot", "mx_dot", "my_dot", "mz_dot"]
# runs from before the two-engine model (19 states) don't match these names and are printed raw


def print_labeled(vector, names):
    """Print a vector as name = value pairs, if the names fit."""
    if len(vector) != len(names):
        print(f"  {vector}")
        return
    for name, value in zip(names, vector):
        print(f"  {name:>6} = {value: .6g}")

def print_trajectory(array, names):
    """Print first and last row of a trajectory plus min / max per column."""
    if array.shape[1] != len(names):
        print(array)
        return
    print(f"  {'':>6}{'first':>13}{'last':>13}{'min':>13}{'max':>13}")
    for ii, name in enumerate(names):
        col = array[:, ii]
        print(f"  {name:>6}{col[0]:>13.5g}{col[-1]:>13.5g}{col.min():>13.5g}{col.max():>13.5g}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python inspect_npz.py <file.npz> [--full]")
        sys.exit(1)

    path = sys.argv[1]
    full = "--full" in sys.argv     # print every value of every array

    data = np.load(path)
    print(f"File: {path}")
    print(f"Arrays: {data.files}")

    for key in data.files:
        array = data[key]
        print()
        print(f"{key}: shape {array.shape}, dtype {array.dtype}")

        if full:
            with np.printoptions(threshold=sys.maxsize, linewidth=200):
                print(array)
        elif key in ("s0", "s_goal"):
            print_labeled(array, STATE_NAMES)
        elif key == "S_opt":
            print_trajectory(array, STATE_NAMES)
        elif key == "U_opt":
            print_trajectory(array, CONTROL_NAMES)
        else:
            print(array)
