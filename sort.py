import sys

import numpy as np
import polars as pl
from llvmlite.binding.targets import os
from numba import njit
from numpy.typing import NDArray

from utils import load


@njit
def compute_indptr_serial(sources: NDArray[np.int32], N: int):
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

    # Sort edges by source (primary) then target (secondary) for CSR representation
    df = pl.DataFrame({"sources": sources, "targets": targets})

    print("Sorting edges by (sources, targets)")
    df = df.sort(["sources", "targets"])

    print("Computing CSR indptr...")
    csr_indptr = compute_indptr_serial(df["sources"].to_numpy(), len(ids))
    csr_indices = df["targets"].to_numpy()

    print("Saving edges-csr.npz...")
    np.savez(
        os.path.join(directory, "edges-csr.npz"),
        indptr=csr_indptr,
        indices=csr_indices,
    )

    print("Sorting edges by (targets, sources)")
    df = df.sort(["targets", "sources"])

    print("Computing CSC indptr...")
    csc_indptr = compute_indptr_serial(df["targets"].to_numpy(), len(ids))
    csc_indices = df["sources"].to_numpy()

    print("Saving edges-csc.npz...")
    np.savez(
        os.path.join(directory, "edges-csc.npz"),
        indptr=csc_indptr,
        indices=csc_indices,
    )
