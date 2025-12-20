import sys

import numpy as np
import polars as pl
import pyarrow as pa
import vortex as vx
from llvmlite.binding.targets import os
from numba import njit
from numpy.typing import NDArray

from utils import load


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

    print("Sorting edges by (targets, sources)")
    df = df.sort(["targets", "sources"])

    print("Saving edges-csc-indices.vortex")
    csc_indices = df["sources"].to_numpy()
    vx.io.write(
        vx.Array.from_arrow(pa.array(csc_indices, type=pa.int32())),
        os.path.join(directory, "edges-csc-indices.vortex"),
    )

    print("Computing CSC indptr...")
    csc_indptr = compute_indptr_serial(df["targets"].to_numpy(), len(ids))

    print("Saving edges-csc-indptr.vortex")
    vx.io.write(
        vx.Array.from_arrow(pa.array(csc_indptr, type=pa.int64())),
        os.path.join(directory, "edges-csc-indptr.vortex"),
    )
