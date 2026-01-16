import os
import sys
import warnings
from typing import Literal

import numba
import numpy as np
import tqdm
from dotenv import load_dotenv
from numba import jit
from numpy.typing import NDArray

from utils import load, save

load_dotenv()


@jit(nopython=True, nogil=True, fastmath=True)
def update_wgrad_clipped(learning_rate: float, loss: float, w1, w2):
    """
    Update w1 toward/away from w2, clamped in unit sphere
    """
    for k in range(w1.size):
        grad = loss * w2[k]
        w1[k] = w1[k] - learning_rate * grad
        if w1[k] < -1.0:
            w1[k] = -1.0
        elif w1[k] > 1.0:
            w1[k] = 1.0


@jit(nopython=True, nogil=True, fastmath=True)
def sample_neighbor_weighted(
    indptr: NDArray[np.int64],
    indices: NDArray[np.int32],
    alias_probs: NDArray[np.uint16],
    alias_indices: NDArray[np.int32],
    node: int,
) -> int:
    """Sample a neighbor with precomputed alias table weighting."""
    start = indptr[node]
    end = indptr[node + 1]
    degree = end - start

    if degree == 0:
        return -1

    # Pick random slot
    slot = np.random.randint(0, degree)
    idx = start + slot

    # Alias sampling
    if np.random.randint(0, 0x10000) < alias_probs[idx]:
        return int(indices[idx])
    else:
        return int(alias_indices[idx])


@jit(nopython=True, nogil=True, fastmath=True)
def sample_neighbor(
    indptr: NDArray[np.int64], indices: NDArray[np.int32], node: int
) -> int:
    """
    Sample a random neighbor of node from CSR/CSC structure. Returns -1 if no neighbors.
    """
    start = indptr[node]
    end = indptr[node + 1]
    degree = end - start
    if degree == 0:
        return -1
    idx = np.random.randint(0, degree)
    return int(indices[start + idx])


@jit(nopython=True, nogil=True, fastmath=True)
def get_degree(indptr: NDArray[np.int64], node: int) -> int:
    """
    Get degree of node from CSR/CSC indptr.
    """
    return indptr[node + 1] - indptr[node]


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def nnvec_edges_update(
    n_nodes: int,
    csr_indices: NDArray[np.int32],
    csr_indptr: NDArray[np.int64],
    csr_alias_probs: NDArray[np.uint16],
    csr_alias_indices: NDArray[np.int32],
    csc_indices: NDArray[np.int32],
    csc_indptr: NDArray[np.int64],
    csc_alias_probs: NDArray[np.uint16],
    csc_alias_indices: NDArray[np.int32],
    # mutual_sources: NDArray[np.int32],
    # mutual_targets: NDArray[np.int32],
    mutual_edges: NDArray[np.int32],
    mutual_degrees: NDArray[np.uint32],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    learning_rate=0.01,
    exponent=0.5,
    max_loss=10.0,
) -> float:
    """
    Euclidean metric (dot product) version.

    This implementation is UNSAFE.
    We concurrently write to weights and gradients in separate threads
    This is only saved by the fact that edges >>> threads
    so pr(race condition) is very low
    """
    # (n_edges,) = weights.shape
    # (src, dst) = coords
    total_loss = 0.0

    n_mutuals = len(mutual_edges)
    for edge in numba.prange(n_mutuals):
        node2, node1 = mutual_edges[edge]
        # node2 = mutual_targets[edge]
        # node1 = mutual_sources[edge]
        # Loss is dot product b/w two connected nodes
        pred = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        loss = pred - 1.0
        # Clip the loss for numerical stability.
        if loss < -max_loss:
            loss = -max_loss
        elif loss > max_loss:
            loss = max_loss
        # Update weights
        update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        # Update biases
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss
        # track losses for early stopping
        total_loss = total_loss + np.abs(loss)

    min_degree = 16
    for A in numba.prange(n_nodes):
        mutual_degree = mutual_degrees[A]
        if mutual_degree < min_degree:
            for _ in range(min_degree - mutual_degree):
                # Co-follower attraction: A follows T, B also follows T
                # Choose T with weight inversely proportional to incoming degree

                # T = sample_neighbor(csr_indptr, csr_indices, A)
                T = sample_neighbor_weighted(
                    csr_indptr, csr_indices, csr_alias_probs, csr_alias_indices, A
                )

                if T != -1:
                    # B = sample_neighbor(csc_indptr, csc_indices, T)
                    B = sample_neighbor_weighted(
                        csc_indptr, csc_indices, csc_alias_probs, csc_alias_indices, T
                    )

                    if B != -1 and B != A:
                        pred = np.dot(w[A], w[B])
                        loss = pred - 1.0
                        if loss < -max_loss:
                            loss = -max_loss
                        elif loss > max_loss:
                            loss = max_loss
                        update_wgrad_clipped(learning_rate, loss, w[A], w[B])
                        total_loss += np.abs(loss)

                # Co-followed attraction: S follows A, S also follows B
                # Choose S with weight inversely proportional to outgoing degree

                # S = sample_neighbor(csc_indptr, csc_indices, A)
                S = sample_neighbor_weighted(
                    csc_indptr, csc_indices, csc_alias_probs, csc_alias_indices, A
                )

                if S != -1:
                    # B = sample_neighbor(csr_indptr, csr_indices, S)
                    B = sample_neighbor_weighted(
                        csr_indptr, csr_indices, csr_alias_probs, csr_alias_indices, S
                    )

                    if B != -1 and B != A:
                        pred = np.dot(w[A], w[B])
                        loss = pred - 1.0
                        if loss < -max_loss:
                            loss = -max_loss
                        elif loss > max_loss:
                            loss = max_loss
                        update_wgrad_clipped(learning_rate, loss, w[A], w[B])
                        total_loss += np.abs(loss)

    return total_loss / n_mutuals


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def nnvec_reverse(
    n_edges: int,
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    learning_rate=0.01,
    max_loss=10.0,
):
    """
    Negative sampling NNVec pass (Euclidean/dot product version)

    negative_edges : array shaped [n_samples, 2]
        We pass in here to hoist out the complexity of
        handling multithreaded RNG (which plays poorly with numba)
    """
    nnodes = w.shape[0]
    for _ in numba.prange(n_edges):
        # TODO: this thrashes the cache. Find a clever soln
        node1 = np.random.randint(0, nnodes)
        node2 = np.random.randint(0, nnodes)
        # We assume no edge (weight = 0) between nodes on negative sampling pass
        loss = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        loss = min(loss, max_loss)
        loss = max(loss, -max_loss)
        update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss


###########################
#                         #
#    /\ Contraction pass  #
#    ||                   #
#    \/ Relaxation pass   #
#                         #
###########################


##########################
#                        #
#       Main method      #
#                        #
##########################


def nnvec_main(
    n_nodes: int,
    csr_indices: NDArray[np.int32],
    csr_indptr: NDArray[np.int64],
    csr_alias_probs: NDArray[np.uint16],
    csr_alias_indices: NDArray[np.int32],
    csc_indices: NDArray[np.int32],
    csc_indptr: NDArray[np.int64],
    csc_alias_probs: NDArray[np.uint16],
    csc_alias_indices: NDArray[np.int32],
    mutual_edges: NDArray[np.int32],
    # mutual_sources: NDArray[np.int32],
    # mutual_targets: NDArray[np.int32],
    mutual_degrees: NDArray[np.uint32],
    n_components: int,
    learning_rate=0.05,
    tol: Literal["auto"] | float = "auto",
    tol_samples=75,
    negative_ratio=0.15,
    negative_decay=0.0,
    exponent=0.5,
    max_loss=30.0,
    max_epoch=500,
):
    """
    NNVec: Fast global first (and higher) order local embeddings.

    This algorithm directly minimizes related nodes' distances.
    It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
    To find stable embeddings based on the minimal dot product of edge weights.

    Parameters:
    -------------
    G: scipy.sparse.coo_array
    n_components (int):
        Number of individual embedding dimensions.
    negative_ratio : float in [0, 1]
        Negative sampling ratio.
        Setting this higher will do more negative sampling.
        This is slower, but can lead to higher quality embeddings.
    exponent : float
        Weighing exponent in loss function.
        Having this lower reduces effect of large edge weights.
    tol : float in [0, 1] or "auto"
        Optimization early stopping criterion.
        Stops average loss < tol for tol_samples epochs.
        "auto" sets tol as a function of learning_rate
    tol_samples : int
        Optimization early stopping criterion.
        This is the number of epochs to sample for loss stability.
        Once loss is stable over this number of epochs we stop early.
    negative_decay : float in [0, 1]
        Decay on negative ratio.
        If >0 then negative ratio will decay by (1-negative_decay) ** epoch
        You should usually leave this to 0.
    max_epoch : int
        Stopping criterion.
    max_count : int
        Ceiling value on edge weights for numerical stability
    learning_rate : float in [0, 1]
        Optimization learning rate.
    max_loss : float
        Loss value ceiling for numerical stability.
    """
    if tol == "auto":
        tol = max(learning_rate / 2, 0.05)

    w: NDArray[np.float32] = (
        np.random.rand(n_nodes, n_components).astype(np.float32) - 0.5
    )
    b: NDArray[np.float32] = np.zeros(n_nodes, dtype=np.float32)
    latest_loss = [np.inf] * tol_samples
    loss = float("inf")
    epoch_range = tqdm.trange(0, max_epoch)
    for epoch in epoch_range:
        # Relaxation pass
        # Number of negative edges
        # neg_edges = int(nnodes * negative_ratio * ((1 - negative_decay) ** epoch))
        neg_edges = int(
            len(mutual_edges) * negative_ratio * ((1 - negative_decay) ** epoch)
        )

        nnvec_reverse(neg_edges, w, b, learning_rate=learning_rate, max_loss=max_loss)
        # Positive "contraction" pass
        # TODO: return only loss max
        loss = nnvec_edges_update(
            n_nodes=n_nodes,
            csr_indices=csr_indices,
            csr_indptr=csr_indptr,
            csr_alias_probs=csr_alias_probs,
            csr_alias_indices=csr_alias_indices,
            csc_indices=csc_indices,
            csc_indptr=csc_indptr,
            csc_alias_probs=csc_alias_probs,
            csc_alias_indices=csc_alias_indices,
            mutual_edges=mutual_edges,
            # mutual_sources=mutual_sources,
            # mutual_targets=mutual_targets,
            mutual_degrees=mutual_degrees,
            w=w,
            b=b,
            learning_rate=learning_rate,
            exponent=exponent,
            max_loss=max_loss,
        )
        # Pct Change in loss
        max_latest = np.max(latest_loss)
        min_latest = np.min(latest_loss)
        if (epoch > tol_samples) and (
            np.abs((max_latest - min_latest) / max_latest) < tol
        ):
            if loss < max_loss:
                print(f"[Loss: {loss:.4f}]: Converged!")
                return w
            else:
                err_str = (
                    f"Could not learn: loss {loss} = max loss {max_loss}\n"
                    + "This is often due to too large learning rates."
                )
                print(err_str)
                warnings.warn(err_str, stacklevel=1)
                break
        elif not np.isfinite(loss).all():
            raise ValueError(
                f"non finite loss: {latest_loss} on epoch {epoch}\n"
                + f"Losses: {loss}\n"
                + f"Previous losses: {[x for x in latest_loss if np.isfinite(x)]}"
                + f"Try reducing the learning rate"
            )
        else:
            latest_loss.append(loss)
            latest_loss = latest_loss[1:]
            epoch_range.set_description(f"[Loss: {loss:.4f}]")
    warnings.warn(f"GVec has not converged. Loss: {loss:.4f}", stacklevel=1)
    return w


if __name__ == "__main__":
    assert "DIM" in os.environ

    dim = int(os.environ["DIM"])

    # Build kwargs for nnvec parameters from environment variables
    nnvec_kwargs = {}
    if "LEARNING_RATE" in os.environ:
        nnvec_kwargs["learning_rate"] = float(os.environ["LEARNING_RATE"])
    if "NEGATIVE_RATIO" in os.environ:
        nnvec_kwargs["negative_ratio"] = float(os.environ["NEGATIVE_RATIO"])
    if "NEGATIVE_DECAY" in os.environ:
        nnvec_kwargs["negative_decay"] = float(os.environ["NEGATIVE_DECAY"])
    if "MAX_LOSS" in os.environ:
        nnvec_kwargs["max_loss"] = float(os.environ["MAX_LOSS"])
    if "MAX_EPOCH" in os.environ:
        nnvec_kwargs["max_epoch"] = int(os.environ["MAX_EPOCH"])
    if "TOL" in os.environ:
        nnvec_kwargs["tol"] = float(os.environ["TOL"])
    if "TOL_SAMPLES" in os.environ:
        nnvec_kwargs["tol_samples"] = int(os.environ["TOL_SAMPLES"])

    if "N_THREADS" in os.environ:
        numba.set_num_threads(int(os.environ["N_THREADS"]))

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    csr_indices: NDArray[np.int32] = load(directory, "edges-csr-indices.npy")
    csr_indptr: NDArray[np.int64] = load(directory, "edges-csr-indptr.npy")
    csr_alias_probs: NDArray[np.uint16] = load(directory, "edges-csr-alias-probs.npy")
    csr_alias_indices: NDArray[np.int32] = load(
        directory, "edges-csr-alias-indices.npy"
    )

    csc_indices: NDArray[np.int32] = load(directory, "edges-csc-indices.npy")
    csc_indptr: NDArray[np.int64] = load(directory, "edges-csc-indptr.npy")
    csc_alias_probs: NDArray[np.uint16] = load(directory, "edges-csc-alias-probs.npy")
    csc_alias_indices: NDArray[np.int32] = load(
        directory, "edges-csc-alias-indices.npy"
    )

    mutual_edges: NDArray[np.int32] = load(directory, "mutual-edges-coo.npy")
    mutual_degrees: NDArray[np.uint32] = load(directory, "mutual-degrees.npy")

    embeddings = nnvec_main(
        len(mutual_degrees),
        csr_indices=csr_indices,
        csr_indptr=csr_indptr,
        csr_alias_probs=csr_alias_probs,
        csr_alias_indices=csr_alias_indices,
        csc_indices=csc_indices,
        csc_indptr=csc_indptr,
        csc_alias_probs=csc_alias_probs,
        csc_alias_indices=csc_alias_indices,
        # mutual_sources=mutual_sources,
        # mutual_targets=mutual_targets,
        mutual_edges=mutual_edges,
        mutual_degrees=mutual_degrees,
        n_components=dim,
        # max_epoch=500,
        # tol_samples=500,
        **nnvec_kwargs,
    )

    save(directory, f"embeddings-{dim}.npy", embeddings)
