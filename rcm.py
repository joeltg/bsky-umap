"""
Reverse Cuthill-McKee ordering for sparse symmetric graphs.

Author: Paul Nation  -- <nonhermitian@gmail.com>
Original Source: QuTiP: Quantum Toolbox in Python (qutip.org)
License: New BSD, (C) 2014

Rewritten in pure Python for symmetric CSR format with Numba optimization.
"""

import sys

import numpy as np
from numba import jit, njit, prange
from numpy.typing import NDArray

from utils import load, save


@jit(nopython=True, parallel=True, cache=True)
def _node_degrees(
    indices: NDArray[np.int32], indptr: NDArray[np.int64], num_rows: int
) -> NDArray[np.uint32]:
    """
    Find the degree of each node (matrix row) in a symmetric graph
    represented by CSR format (indices, indptr).

    Parameters
    ----------
    indices : ndarray
        CSR indices array
    indptr : ndarray
        CSR indptr array
    num_rows : int
        Number of rows in the matrix

    Returns
    -------
    degree : ndarray
        Degree of each node
    """
    degree = np.zeros(num_rows, dtype=np.uint32)

    for i in prange(num_rows):
        degree[i] = indptr[i + 1] - indptr[i]
        # Check if diagonal is present and add one if so
        for j in range(indptr[i], indptr[i + 1]):
            if indices[j] == i:
                degree[i] += 1
                break

    return degree


@jit(nopython=True, cache=True)
def _reverse_cuthill_mckee_impl(
    indices: NDArray[np.int32],
    indptr: NDArray[np.int64],
    num_rows: int,
    degree: NDArray[np.uint32],
    inds: NDArray[np.intp],
    rev_inds: NDArray[np.intp],
) -> NDArray[np.uint32]:
    """
    Core RCM implementation with Numba optimization.

    Parameters
    ----------
    indices : ndarray
        CSR indices array
    indptr : ndarray
        CSR indptr array
    num_rows : int
        Number of rows
    degree : ndarray
        Precomputed node degrees
    inds : ndarray
        Sorted indices by degree (will be modified)
    rev_inds : ndarray
        Reverse lookup for inds

    Returns
    -------
    order : ndarray
        Permutation array in RCM order
    """
    # Array to store the ordering
    order = np.zeros(num_rows, dtype=np.uint32)

    # Temporary array for insertion sort
    max_degree = np.max(degree) if num_rows > 0 else 0
    temp_degrees = np.zeros(max_degree, dtype=np.uint32)
    temp_nodes = np.zeros(max_degree, dtype=np.uint32)

    N = 0  # Current position in order array

    # Loop over all nodes to handle disconnected components
    for zz in range(num_rows):
        if inds[zz] != -1:  # Start new BFS from unvisited node with lowest degree
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1

            level_start = N - 1
            level_end = N

            # BFS traversal
            while level_start < level_end:
                # Process all nodes at current level
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N

                    # Add unvisited neighbors
                    for jj in range(indptr[i], indptr[i + 1]):
                        j = indices[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1

                    # Sort newly added nodes by degree (insertion sort)
                    level_len = N - N_old

                    # Copy degrees and nodes to temporary arrays
                    for kk in range(level_len):
                        temp_degrees[kk] = degree[order[N_old + kk]]
                        temp_nodes[kk] = order[N_old + kk]

                    # Insertion sort
                    for kk in range(1, level_len):
                        temp_deg = temp_degrees[kk]
                        temp_node = temp_nodes[kk]
                        ll = kk

                        while ll > 0 and temp_deg < temp_degrees[ll - 1]:
                            temp_degrees[ll] = temp_degrees[ll - 1]
                            temp_nodes[ll] = temp_nodes[ll - 1]
                            ll -= 1

                        temp_degrees[ll] = temp_deg
                        temp_nodes[ll] = temp_node

                    # Copy sorted nodes back
                    for kk in range(level_len):
                        order[N_old + kk] = temp_nodes[kk]

                # Move to next level
                level_start = level_end
                level_end = N

        if N == num_rows:
            break

    # Return reversed order for RCM
    return order[::-1]


def reverse_cuthill_mckee(
    indices: NDArray[np.int32], indptr: NDArray[np.int64]
) -> NDArray[np.uint32]:
    """
    Returns the permutation array that orders a sparse symmetric CSR matrix
    in Reverse-Cuthill McKee ordering.

    This implementation assumes the input graph is symmetric (or that you're
    passing the symmetrized version). It operates directly on CSR format
    (indices, indptr) tuples without weights.

    Parameters
    ----------
    indices : ndarray
        CSR indices array
    indptr : ndarray
        CSR indptr array

    Returns
    -------
    perm : ndarray
        Array of permuted row and column indices in RCM order.

    Notes
    -----
    The algorithm performs BFS starting from nodes of lowest degree for each
    connected component, following the original Cuthill-McKee paper.

    This implementation uses Numba JIT compilation for performance. The first
    call will compile the functions, so subsequent calls will be much faster.

    References
    ----------
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric Matrices",
    ACM '69 Proceedings of the 1969 24th national conference, (1969).

    Examples
    --------
    >>> indices = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=np.uint32)
    >>> indptr = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    >>> perm = reverse_cuthill_mckee(indices, indptr)
    >>> print(perm)
    """
    # Ensure arrays are numpy arrays with proper dtypes
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)

    num_rows = len(indptr) - 1

    # Compute node degrees (parallelized)
    degrees = _node_degrees(indices, indptr, num_rows)

    # Sort nodes by degree (ascending)
    inds = np.argsort(degrees).copy()
    rev_inds = np.argsort(inds)

    # Run the main RCM algorithm
    return _reverse_cuthill_mckee_impl(
        indices, indptr, num_rows, degrees, inds, rev_inds
    )


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

    ids: NDArray[np.uint32] = load(directory, "ids.npy", copy=True)
    edges: NDArray[np.int32] = load(directory, "mutual-edges-coo.npy")

    indices = edges[:, 1]
    indptr = compute_indptr_serial(edges[:, 0], len(ids))

    perm = reverse_cuthill_mckee(indices, indptr)

    save(directory, "id-rcm-perm.npy", perm)
    save(directory, "ids-rcm.npy", ids[perm])
