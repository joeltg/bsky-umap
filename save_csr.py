import sys

import numpy as np
import polars as pl
from numba import njit
from numpy.typing import NDArray

from utils import load, save


@njit
def compute_indptr_serial(sources: NDArray[np.int32], N: int) -> NDArray[np.int64]:
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

    ids: NDArray[np.uint32] = load(directory, "ids.npy")

    sources: NDArray[np.int32] = load(directory, "sources.npy")
    targets: NDArray[np.int32] = load(directory, "targets.npy")

    df = pl.DataFrame({"sources": sources, "targets": targets})

    print("Sorting edges by (sources, targets)")
    df = df.sort(["sources", "targets"])

    csr_indices = df["targets"].to_numpy()
    save(directory, "edges-csr-indices.npy", csr_indices)

    print("Computing CSR indptr...")
    csr_indptr = compute_indptr_serial(df["sources"].to_numpy(), len(ids))

    save(directory, "edges-csr-indptr.npy", csr_indptr)
