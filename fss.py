import os
import sys
from typing import cast

import numba
import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from utils import KNNReader, write_edges
from zumap.zumap import fuzzy_simplicial_set

load_dotenv()


def main():
    dim = int(os.environ["DIM"])
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    n_threads = int(os.environ["N_THREADS"])

    numba.set_num_threads(n_threads)

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    knn_path = os.path.join(directory, f"knn-{dim}-{n_neighbors}.arrow")
    with KNNReader(knn_path, n_neighbors) as reader:
        neighbor_graph = reader.get_knn()

    (knn_indices, knn_dists) = neighbor_graph
    set_op_mix_ratio = 1.0
    local_connectivity = 1.0

    print(f"got knn indices with shape {knn_indices.shape}")
    print(f"got knn dists with shape {knn_dists.shape}")

    fss = fuzzy_simplicial_set(
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
    )

    (rows, cols) = fss.coords
    assert fss.data.dtype == np.float32
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32

    path = os.path.join(directory, f"fss-{dim}-{n_neighbors}.arrow")

    edges = (
        cast(NDArray[np.float32], fss.data),
        cast(NDArray[np.int32], rows),
        cast(NDArray[np.int32], cols),
    )

    write_edges(path, edges)


if __name__ == "__main__":
    main()
