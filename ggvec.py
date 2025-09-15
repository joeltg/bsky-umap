import warnings
from typing import Literal

import numba
import numpy as np
import tqdm
from numba import jit
from numpy.typing import NDArray


@jit(nopython=True, nogil=True, fastmath=True)
def _update_wgrad_clipped(learning_rate: float, loss: float, w1, w2):
    """same as above, clamped in unit sphere"""
    for k in range(w1.size):
        grad = loss * w2[k]
        w1[k] = w1[k] - learning_rate * grad
        if w1[k] < -1.0:
            w1[k] = -1.0
        elif w1[k] > 1.0:
            w1[k] = 1.0


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _ggvec_edges_update(
    G: tuple[int, NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32]],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    learning_rate=0.01,
    exponent=0.5,
    max_loss=10.0,
):
    """
    This implementation is UNSAFE.
    We concurrently write to weights and gradients in separate threads
    This is only saved by the fact that edges >>> threads
    so pr(race condition) is very low

    Couple of issues:
        - Only one weight matrix
        - unvectorized
        - unsafe
        - Assumes symmetric edges (would need two w matrix for directed graphs)
    Implementation inspired from https://github.com/maciejkula/glove-python/blob/master/glove/glove_cython.pyx
    """
    (nnodes, src, dst, weights) = G
    n_edges = len(weights)
    total_loss = 0.0
    for edge in numba.prange(n_edges):
        node2 = dst[edge]
        node1 = src[edge]
        # Loss is dot product b/w two connected nodes
        pred = np.dot(w[node1], w[node2]) + b[node1] + b[node2]
        # Use arcsinh(weight) instead of log(weight)
        #     handles all real valued weights well
        loss = pred - weights[edge] ** exponent
        # Clip the loss for numerical stability.
        if loss < -max_loss:
            loss = -max_loss
        elif loss > max_loss:
            loss = max_loss
        # Update weights
        _update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        _update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        # Update biases
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss
        # track losses for early stopping
        total_loss = total_loss + np.abs(loss)
    return total_loss / n_edges


###########################
#                         #
#    /\ Contraction pass  #
#    ||                   #
#    \/ Relaxation pass   #
#                         #
###########################


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _ggvec_reverse(
    n_edges: int,
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    learning_rate=0.01,
    max_loss=10.0,
):
    """
    Negative sampling GGVec pass

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
        _update_wgrad_clipped(learning_rate, loss, w[node1], w[node2])
        _update_wgrad_clipped(learning_rate, loss, w[node2], w[node1])
        b[node1] -= learning_rate * loss
        b[node2] -= learning_rate * loss


##########################
#                        #
#       Main method      #
#                        #
##########################


def ggvec_main(
    G: tuple[int, NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32]],
    n_components=2,
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
    GGVec: Fast global first (and higher) order local embeddings.

    This algorithm directly minimizes related nodes' distances.
    It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
    To find stable embeddings based on the minimal dot product of edge weights.

    Parameters:
    -------------
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
    (nnodes, sources, targets, weights) = G

    w: NDArray[np.float32] = (
        np.random.rand(nnodes, n_components).astype(np.float32) - 0.5
    )
    b: NDArray[np.float32] = np.zeros(nnodes, dtype=np.float32)
    latest_loss = [np.inf] * tol_samples
    loss = float("inf")
    epoch_range = tqdm.trange(0, max_epoch)
    for epoch in epoch_range:
        # Relaxation pass
        # Number of negative edges
        # neg_edges = int(nnodes * negative_ratio * ((1 - negative_decay) ** epoch))
        neg_edges = int(len(weights) * negative_ratio * ((1 - negative_decay) ** epoch))
        _ggvec_reverse(neg_edges, w, b, learning_rate=learning_rate, max_loss=max_loss)
        # Positive "contraction" pass
        # TODO: return only loss max
        loss = _ggvec_edges_update(
            G,
            w,
            b,
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
