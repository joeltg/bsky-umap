import warnings

import numba
import numpy as np
import tqdm
from numba import jit


@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def _dot_product_simd(w1, w2):
    """SIMD-friendly dot product"""
    sum = np.float32(0.0)
    for k in range(w1.size):
        sum += w1[k] * w2[k]
    return sum


@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def _update_embedding_fused(lr_loss, w1, w2):
    """Fused multiply-subtract-clamp operation"""
    for k in range(w1.size):
        # Update w1 with gradient from w2
        grad = lr_loss * w2[k]
        new_val = w1[k] - grad
        # Branchless clamping
        w1[k] = min(max(new_val, np.float32(-1.0)), np.float32(1.0))


@jit(nopython=True, nogil=True, fastmath=True)
def _fast_random_pair(nnodes, state):
    """Linear congruential generator for fast random pairs"""
    # Update state
    state[0] = (state[0] * np.uint32(1664525) + np.uint32(1013904223)) & np.uint32(
        0xFFFFFFFF
    )
    node1 = (state[0] >> np.uint32(16)) % nnodes
    # Generate second number
    state[0] = (state[0] * np.uint32(1664525) + np.uint32(1013904223)) & np.uint32(
        0xFFFFFFFF
    )
    node2 = (state[0] >> np.uint32(16)) % nnodes
    return node1, node2


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _ggvec_edges_update_batched(
    src, dst, data, w, b, learning_rate=0.01, exponent=0.5, max_loss=10.0, batch_size=64
):
    """
    Batched edge updates with SIMD dot products and fused operations.

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
    n_edges = dst.size
    # scale_factor = np.sqrt(w.shape[0])
    scale_factor = np.sqrt(np.float32(w.shape[1]))
    inv_scale_factor = np.float32(1.0) / scale_factor

    n_batches = (n_edges + batch_size - 1) // batch_size

    total_loss = np.float32(0.0)

    for batch_idx in numba.prange(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_edges)

        for edge in range(batch_start, batch_end):
            node1 = dst[edge]
            node2 = src[edge]

            # SIMD dot product with scaling
            pred = (
                _dot_product_simd(w[node1], w[node2]) + b[node1] + b[node2]
            ) * inv_scale_factor

            # Compute loss with edge weight transformation
            target = data[edge] ** exponent
            loss = pred - target

            # Branchless clamping
            loss = min(max(loss, -max_loss), max_loss)

            # Fused updates
            lr_loss = learning_rate * loss
            _update_embedding_fused(lr_loss, w[node1], w[node2])
            _update_embedding_fused(lr_loss, w[node2], w[node1])

            # Update biases
            b[node1] -= lr_loss
            b[node2] -= lr_loss

            # Accumulate loss (race condition acceptable for monitoring)
            total_loss += np.abs(loss)

    return total_loss / np.float32(n_edges)


###########################
#                         #
#    /\ Contraction pass  #
#    ||                   #
#    \/ Relaxation pass   #
#                         #
###########################


@jit(nopython=True, nogil=True, fastmath=True, parallel=True)
def _ggvec_reverse_batched(
    n_edges, w, b, learning_rate=0.01, max_loss=10.0, batch_size=64
):
    """
    Negative sampling with fast RNG and batched updates
    """
    nnodes = w.shape[0]
    # scale_factor = np.sqrt(w.shape[0])
    scale_factor = np.sqrt(np.float32(w.shape[1]))
    inv_scale_factor = np.float32(1.0) / scale_factor
    n_batches = (n_edges + batch_size - 1) // batch_size

    # Parallelize over batches
    for batch_idx in numba.prange(n_batches):
        # Each thread gets its own RNG state
        thread_id = np.uint32(numba.get_thread_id())
        rng_state = np.zeros(1, dtype=np.uint32)
        rng_state[0] = (
            np.uint32(1337) + thread_id * np.uint32(17) + batch_idx * np.uint32(31)
        )

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_edges)

        for _ in range(batch_start, batch_end):
            # Fast random pair generation
            node1, node2 = _fast_random_pair(nnodes, rng_state)

            # We assume no edge (weight = 0) between nodes on negative sampling pass
            loss = (
                _dot_product_simd(w[node1], w[node2]) + b[node1] + b[node2]
            ) * inv_scale_factor

            # Branchless clamping
            loss = min(max(loss, -max_loss), max_loss)

            # Fused updates
            lr_loss = learning_rate * loss
            _update_embedding_fused(lr_loss, w[node1], w[node2])
            _update_embedding_fused(lr_loss, w[node2], w[node1])

            # Update biases
            b[node1] -= lr_loss
            b[node2] -= lr_loss


##########################
#                        #
#       Main method      #
#                        #
##########################


def ggvec_main(
    src,
    dst,
    data,
    n_nodes,
    n_components=2,
    learning_rate=0.05,
    tol=0.005,
    tol_samples=75,
    negative_ratio=0.15,
    negative_decay=0.0,
    exponent=0.5,
    max_loss=30.0,
    max_epoch=500,
    batch_size=64,
    n_threads=None,
):
    """
    GGVec: Fast global first (and higher) order local embeddings.

    This algorithm directly minimizes related nodes' distances.
    It uses a relaxation pass (negative sample) + contraction pass (loss minimization)
    To find stable embeddings based on the minimal dot product of edge weights.

    Parameters:
    -------------
    src : array of int
        Source node indices for each edge.
    dst : array of int
        Destination node indices for each edge.
    data : array of float
        Edge weights.
    n_nodes : int
        Total number of nodes in the graph.
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
    learning_rate : float in [0, 1]
        Optimization learning rate.
    max_loss : float
        Loss value ceiling for numerical stability.
    batch_size : int
        Batch size for positive and negative sampling
    n_threads : int, optional
        Maximum number of threads to use for parallel computation.
        If None, uses all available threads. Thread count is restored
        to original value after completion.
    """
    # Store original thread count and set new one if specified
    original_threads = numba.get_num_threads()
    if n_threads is not None:
        numba.set_num_threads(min(original_threads, n_threads))

    try:
        # Ensure inputs are float32
        data = data.astype(np.float32)

        nnodes = n_nodes
        w = np.random.rand(nnodes, n_components).astype(np.float32) - np.float32(0.5)
        b = np.zeros(nnodes, dtype=np.float32)

        latest_loss = [np.float32(np.inf)] * tol_samples
        loss = np.float32(np.inf)

        epoch_range = tqdm.trange(0, max_epoch)

        for epoch in epoch_range:
            # Relaxation pass
            # Number of negative edges
            neg_edges = int(dst.size * negative_ratio * ((1 - negative_decay) ** epoch))
            _ggvec_reverse_batched(
                neg_edges,
                w,
                b,
                learning_rate=learning_rate,
                max_loss=max_loss,
                batch_size=batch_size,
            )

            # Positive "contraction" pass
            loss = _ggvec_edges_update_batched(
                src,
                dst,
                data,
                w,
                b,
                learning_rate=learning_rate,
                exponent=exponent,
                max_loss=max_loss,
                batch_size=batch_size,
            )

            # Pct Change in loss
            max_latest = np.max(latest_loss)
            min_latest = np.min(latest_loss)
            if (epoch > tol_samples) and (
                np.abs((max_latest - min_latest) / max_latest) < tol
            ):
                if loss < max_loss:
                    epoch_range.close()
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

    finally:
        # Always restore original thread count
        if n_threads is not None:
            numba.set_num_threads(original_threads)
