import os
import sys

import numba
import numpy as np
import scipy
from dotenv import load_dotenv
from numpy.typing import NDArray

from utils import EdgeReader, load, save
from zumap.zumap import find_ab_params, get_random_state, simplicial_set_embedding

load_dotenv()


def main():
    dim = int(os.environ["DIM"])
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    n_epochs = int(os.environ["N_EPOCHS"])
    n_threads = int(os.environ["N_THREADS"])
    spread = float(os.environ["SPREAD"])
    min_dist = float(os.environ["MIN_DIST"])

    numba.set_num_threads(n_threads)

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    X: NDArray[np.float32] = load(directory, f"embeddings-{dim}.npy")

    fss_path = os.path.join(directory, f"fss-{dim}-{n_neighbors}.arrow")
    with EdgeReader(fss_path) as reader:
        (vals, rows, cols) = reader.get_edges()

    fss = scipy.sparse.coo_array((vals, (rows, cols)))

    random_state = get_random_state(0)
    a, b = find_ab_params(spread, min_dist)

    positions = simplicial_set_embedding(
        X,
        G=fss,
        n_components=2,
        random_state=random_state,
        a=a,
        b=b,
        n_epochs=n_epochs,
        optimize="cpu",
    )

    save(directory, f"positions-{dim}-{n_neighbors}.npy", positions)


if __name__ == "__main__":
    main()
