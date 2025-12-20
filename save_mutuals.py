import sys

import numpy as np
from numba import jit
from numpy.typing import NDArray

from utils import load, load_array, save_array


@jit(nopython=True)
def find_mutual_edges(
    n_nodes: int,
    csr_indices: NDArray[np.int32],
    csr_indptr: NDArray[np.int64],
    csc_indices: NDArray[np.int32],
    csc_indptr: NDArray[np.int64],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Find mutual edges using sorted intersection."""
    # First pass: count mutual edges
    count = 0
    for A in range(n_nodes):
        fol_start = csr_indptr[A]
        fol_end = csr_indptr[A + 1]
        fer_start = csc_indptr[A]
        fer_end = csc_indptr[A + 1]

        # Both arrays are sorted, so we can do merge-style intersection
        i, j = fol_start, fer_start
        while i < fol_end and j < fer_end:
            if csr_indices[i] == csc_indices[j]:
                count += 1
                i += 1
                j += 1
            elif csr_indices[i] < csc_indices[j]:
                i += 1
            else:
                j += 1

    # Second pass: fill arrays
    src = np.empty(count, dtype=np.int32)
    dst = np.empty(count, dtype=np.int32)
    idx = 0
    for A in range(n_nodes):
        fol_start = csr_indptr[A]
        fol_end = csr_indptr[A + 1]
        fer_start = csc_indptr[A]
        fer_end = csc_indptr[A + 1]

        i, j = fol_start, fer_start
        while i < fol_end and j < fer_end:
            if csr_indices[i] == csc_indices[j]:
                src[idx] = A
                dst[idx] = csr_indices[i]
                idx += 1
                i += 1
                j += 1
            elif csr_indices[i] < csc_indices[j]:
                i += 1
            else:
                j += 1

    return src, dst


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    ids: NDArray[np.uint32] = load(directory, "ids.npy")

    csr_indices: NDArray[np.int32] = load_array(directory, "edges-csr-indices.vortex")
    csr_indptr: NDArray[np.int64] = load_array(directory, "edges-csr-indptr.vortex")
    csc_indices: NDArray[np.int32] = load_array(directory, "edges-csc-indices.vortex")
    csc_indptr: NDArray[np.int64] = load_array(directory, "edges-csc-indptr.vortex")

    (sources, targets) = find_mutual_edges(
        len(ids),
        csr_indices=csr_indices,
        csr_indptr=csr_indptr,
        csc_indices=csc_indices,
        csc_indptr=csc_indptr,
    )

    assert sources.shape == targets.shape and sources.dtype == targets.dtype
    print("found mutuals", sources.shape, sources.dtype)

    save_array(directory, "edges-mutual-coo-sources.vortex", sources)
    save_array(directory, "edges-mutual-coo-targets.vortex", targets)
