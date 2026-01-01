import sys

import numpy as np
from numba import jit, numba
from numpy.typing import NDArray

from utils import load, load_array, save_array


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def build_all_alias_tables(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    neighbor_degrees: NDArray[np.uint32],
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Build alias tables for weighted sampling of each node's neighbors.
    Weight is 1/log(1 + degree) of the neighbor.
    """
    n_nodes = len(indptr) - 1
    n_edges = len(indices)
    alias_probs = np.empty(n_edges, dtype=np.float32)
    alias_indices = np.empty(n_edges, dtype=np.int32)

    for A in numba.prange(n_nodes):
        start = indptr[A]
        end = indptr[A + 1]
        degree = end - start

        if degree == 0:
            continue

        if degree == 1:
            alias_probs[start] = np.float16(1.0)
            alias_indices[start] = indices[start]
            continue

        # Allocate working arrays for this node
        weights = np.empty(degree, dtype=np.float32)
        probs = np.ones(degree, dtype=np.float32)
        aliases = np.empty(degree, dtype=np.int32)
        small = np.empty(degree, dtype=np.int32)
        large = np.empty(degree, dtype=np.int32)

        # Compute weights: 1 / log(1 + neighbor_degree)
        total = np.float32(0.0)
        for i in range(degree):
            neighbor = indices[start + i]
            neighbor_deg = neighbor_degrees[neighbor]
            w = np.float32(1.0) / np.log1p(np.float32(neighbor_deg))
            weights[i] = w
            total += w
            aliases[i] = i  # initialize to self

        # Normalize so average = 1
        scale = np.float32(degree) / total
        for i in range(degree):
            weights[i] *= scale

        # Partition into small and large
        n_small = 0
        n_large = 0
        for i in range(degree):
            if weights[i] < 1.0:
                small[n_small] = i
                n_small += 1
            else:
                large[n_large] = i
                n_large += 1

        # Build alias table
        while n_small > 0 and n_large > 0:
            n_small -= 1
            s = small[n_small]
            n_large -= 1
            l = large[n_large]

            probs[s] = weights[s]
            aliases[s] = l

            weights[l] -= np.float32(1.0) - weights[s]

            if weights[l] < 1.0:
                small[n_small] = l
                n_small += 1
            else:
                large[n_large] = l
                n_large += 1

        # Store results with actual node indices
        for i in range(degree):
            alias_probs[start + i] = np.float16(probs[i])
            alias_indices[start + i] = indices[start + aliases[i]]

    return alias_probs, alias_indices


if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    incoming_degrees: NDArray[np.uint32] = load(directory, "incoming_degrees.npy")
    csr_indptr: NDArray[np.int64] = load_array(directory, "edges-csr-indptr.vortex")
    csr_indices: NDArray[np.int32] = load_array(directory, "edges-csr-indices.vortex")

    print("computing csr alias table")
    csr_alias_probs, csr_alias_indices = build_all_alias_tables(
        csr_indptr, csr_indices, incoming_degrees
    )

    csr_alias_probs = csr_alias_probs.astype(np.float16)
    save_array(directory, "edges-csr-alias-probs.vortex", csr_alias_probs)
    save_array(directory, "edges-csr-alias-indices.vortex", csr_alias_indices)

    outgoing_degrees: NDArray[np.uint32] = load(directory, "outgoing_degrees.npy")
    csc_indptr: NDArray[np.int64] = load_array(directory, "edges-csc-indptr.vortex")
    csc_indices: NDArray[np.int32] = load_array(directory, "edges-csc-indices.vortex")

    print("computing csc alias table")
    csc_alias_probs, csc_alias_indices = build_all_alias_tables(
        csc_indptr, csc_indices, outgoing_degrees
    )

    csc_alias_probs = csc_alias_probs.astype(np.float16)
    save_array(directory, "edges-csc-alias-probs.vortex", csc_alias_probs)
    save_array(directory, "edges-csc-alias-indices.vortex", csc_alias_indices)
