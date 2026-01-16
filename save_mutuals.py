import sys

import numpy as np
from numba import jit
from numpy.typing import NDArray

from utils import load, save


@jit(nopython=True)
def find_mutual_edges(
    n_nodes: int,
    csr_indices: NDArray[np.int32],
    csr_indptr: NDArray[np.int64],
    csc_indices: NDArray[np.int32],
    csc_indptr: NDArray[np.int64],
) -> NDArray[np.int32]:
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
    mutuals = np.empty((count, 2), dtype=np.int32)
    idx = 0
    for A in range(n_nodes):
        fol_start = csr_indptr[A]
        fol_end = csr_indptr[A + 1]
        fer_start = csc_indptr[A]
        fer_end = csc_indptr[A + 1]

        i, j = fol_start, fer_start
        while i < fol_end and j < fer_end:
            if csr_indices[i] == csc_indices[j]:
                mutuals[idx][0] = A
                mutuals[idx][1] = csr_indices[i]
                idx += 1
                i += 1
                j += 1
            elif csr_indices[i] < csc_indices[j]:
                i += 1
            else:
                j += 1

    return mutuals


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    csr_indices: NDArray[np.int32] = load(directory, "edges-csr-indices.npy")
    csr_indptr: NDArray[np.int64] = load(directory, "edges-csr-indptr.npy")
    csc_indices: NDArray[np.int32] = load(directory, "edges-csc-indices.npy")
    csc_indptr: NDArray[np.int64] = load(directory, "edges-csc-indptr.npy")

    assert csr_indices.shape == csc_indices.shape
    assert csr_indptr.shape == csc_indptr.shape
    n_nodes = len(csr_indptr) - 1

    mutuals = find_mutual_edges(
        n_nodes,
        csr_indices=csr_indices,
        csr_indptr=csr_indptr,
        csc_indices=csc_indices,
        csc_indptr=csc_indptr,
    )

    print("found mutuals", mutuals.shape)

    outgoing_degrees = np.bincount(mutuals[:, 0], minlength=n_nodes)
    incoming_degrees = np.bincount(mutuals[:, 1], minlength=n_nodes)

    assert np.array_equal(outgoing_degrees, incoming_degrees)
    degrees = outgoing_degrees.astype(np.uint32)

    # upper_triangle_mask = sources < targets

    save(directory, "mutual-edges-coo.npy", mutuals)
    save(directory, "mutual-degrees.npy", degrees)
