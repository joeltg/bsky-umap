import sys

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.sparse import coo_array

from utils import load, save


@njit
def compute_indptr_serial(sources: NDArray[np.uint32], N: int) -> NDArray[np.int64]:
    E = len(sources)
    indptr = np.empty(N + 1, dtype=np.int64)
    edge_idx = 0

    for node in range(N + 1):
        while edge_idx < E and sources[edge_idx] < node:
            edge_idx += 1
        indptr[node] = edge_idx

    return indptr


if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    ids = load(directory, "ids.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    G = coo_array(
        (np.ones(len(sources), dtype=np.float32), (sources, targets)),
        shape=(len(ids), len(ids)),
    ).tocsr()

    save(directory, "csr-data.npy", G.data)
    save(directory, "csr-indices.npy", G.indices)
    save(directory, "csr-indptr.npy", G.indptr)
