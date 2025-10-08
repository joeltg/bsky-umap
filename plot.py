import os
import sys

import matplotlib
import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from utils import load
from zumap import plot

matplotlib.use("GTK4Agg")

load_dotenv()

if __name__ == "__main__":
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    dim = int(os.environ["DIM"])

    metric = os.environ.get("METRIC", "euclidean")

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    positions: NDArray[np.float32] = load(
        directory, f"positions-{dim}-{metric}-{n_neighbors}.npy"
    )

    min_dist = 0.1
    n_neighbors = 10

    axis = plot.points(
        positions, n_neighbors=n_neighbors, min_dist=min_dist, theme="viridis"
    )
    plot.show(axis)
