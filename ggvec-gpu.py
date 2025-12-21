import os
import sys

import numba
import numpy as np
import scipy
import tqdm
from dotenv import load_dotenv
from numba import cuda
from numpy.typing import NDArray

from utils import load, load_coo_array, save

load_dotenv()


# ================================
# Random number generation
# ================================


@cuda.jit(device=True, inline=True)
def splitmix64(state):
    """Splitmix64 RNG. state is a uint64, returns (new_state, random_uint64)."""
    state = state + np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return state, z


@cuda.jit(device=True, inline=True)
def random_uint32(state):
    """Generate a random uint32 and update state."""
    state, z = splitmix64(state)
    return state, np.uint32(z & np.uint64(0xFFFFFFFF))


@cuda.jit(device=True, inline=True)
def random_int_range(state, max_val):
    """Generate random int in [0, max_val). max_val should be int32."""
    state, r = random_uint32(state)
    return state, r % np.uint32(max_val)


# ================================
# Core kernels
# ================================


@cuda.jit
def ggvec_attract_kernel(
    src: NDArray[np.int32],
    dst: NDArray[np.int32],
    weights: NDArray[np.float32],
    embeddings: NDArray[np.float32],
    learning_rate: float,
    max_loss: float,
    loss_per_block: NDArray[np.float32],  # shape: (n_blocks,)
):
    n_edges = src.shape[0]
    n_dims = embeddings.shape[1]

    shared_loss = cuda.shared.array(256, dtype=numba.float32)

    edge_idx = cuda.grid(1)
    tid = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    local_loss = 0.0

    if edge_idx < n_edges:
        # ... compute loss and update embeddings ...
        local_loss = abs(loss)

    # Block reduction
    shared_loss[tid] = local_loss
    cuda.syncthreads()

    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared_loss[tid] += shared_loss[tid + s]
        cuda.syncthreads()
        s //= 2

    # No atomic needed!
    if tid == 0:
        loss_per_block[block_id] = shared_loss[0]


@cuda.jit
def ggvec_repel_kernel(
    n_nodes: int,
    embeddings: NDArray[np.float32],
    learning_rate: float,
    max_loss: float,
    rng_states: NDArray[np.uint64],
):
    """
    Repulsion pass: push random node pairs apart.

    Each thread does one repulsion sample.
    """
    n_dims = embeddings.shape[1]

    thread_idx = cuda.grid(1)
    if thread_idx >= rng_states.shape[0]:
        return

    # Get thread-local RNG state
    state = rng_states[thread_idx]

    # Sample two random nodes
    state, node1 = random_int_range(state, n_nodes)
    state, node2 = random_int_range(state, n_nodes)

    # Save RNG state back
    rng_states[thread_idx] = state

    node1 = np.int32(node1)
    node2 = np.int32(node2)

    if node1 == node2:
        return

    # Compute dot product
    dot = 0.0
    for k in range(n_dims):
        dot += embeddings[node1, k] * embeddings[node2, k]

    # Loss (target is 0 for repulsion)
    loss = dot
    if loss < -max_loss:
        loss = -max_loss
    elif loss > max_loss:
        loss = max_loss

    # Update embeddings with atomics
    for k in range(n_dims):
        grad1 = loss * embeddings[node2, k]
        grad2 = loss * embeddings[node1, k]

        old1 = cuda.atomic.add(embeddings, (node1, k), -learning_rate * grad1)
        new1 = old1 - learning_rate * grad1
        if new1 < -1.0:
            cuda.atomic.add(embeddings, (node1, k), -1.0 - new1)
        elif new1 > 1.0:
            cuda.atomic.add(embeddings, (node1, k), 1.0 - new1)

        old2 = cuda.atomic.add(embeddings, (node2, k), -learning_rate * grad2)
        new2 = old2 - learning_rate * grad2
        if new2 < -1.0:
            cuda.atomic.add(embeddings, (node2, k), -1.0 - new2)
        elif new2 > 1.0:
            cuda.atomic.add(embeddings, (node2, k), 1.0 - new2)


# ================================
# Main function
# ================================


def ggvec_cuda_main(
    n_nodes: int,
    src: NDArray[np.int32],
    dst: NDArray[np.int32],
    weights: NDArray[np.float32],
    n_components: int = 64,
    learning_rate: float = 0.05,
    negative_ratio: float = 0.15,
    max_loss: float = 30.0,
    max_epoch: int = 500,
    threads_per_block: int = 256,
) -> NDArray[np.float32]:
    """
    GPU-accelerated GGVec embedding.

    Parameters:
    -----------
    n_nodes : int
        Number of nodes.
    src, dst : NDArray[np.int32]
        Edge list in COO format.
    weights : NDArray[np.float32]
        Edge weights (typically all 1.0 for unweighted).
    n_components : int
        Embedding dimension.
    learning_rate : float
        SGD learning rate.
    negative_ratio : float
        Ratio of negative samples to edges.
    max_loss : float
        Loss clipping threshold.
    max_epoch : int
        Number of epochs.
    threads_per_block : int
        CUDA threads per block.

    Returns:
    --------
    embeddings : NDArray[np.float32]
        Node embeddings of shape (n_nodes, n_components).
    """
    n_edges = len(src)
    n_neg_samples = int(n_edges * negative_ratio)

    # Initialize embeddings
    embeddings = np.random.rand(n_nodes, n_components).astype(np.float32) - 0.5

    # Initialize RNG states (one per potential thread)
    max_threads = max(n_edges, n_neg_samples)
    rng_states = np.arange(max_threads, dtype=np.uint64) + np.uint64(42)

    # Copy to device
    d_src = cuda.to_device(src)
    d_dst = cuda.to_device(dst)
    d_weights = cuda.to_device(weights)
    d_embeddings = cuda.to_device(embeddings)
    d_rng_states = cuda.to_device(rng_states)

    # Grid dimensions
    attract_blocks = (n_edges + threads_per_block - 1) // threads_per_block
    repel_blocks = (n_neg_samples + threads_per_block - 1) // threads_per_block

    d_loss = cuda.to_device(
        np.zeros(max(attract_blocks, repel_blocks), dtype=np.float32)
    )

    epoch_range = tqdm.trange(max_epoch)
    for epoch in epoch_range:
        # Reset loss accumulator
        d_loss.copy_to_device(np.zeros(1, dtype=np.float32))

        # Repulsion pass
        ggvec_repel_kernel[repel_blocks, threads_per_block](
            n_nodes,
            d_embeddings,
            learning_rate,
            max_loss,
            d_rng_states,
        )

        # Attraction pass
        ggvec_attract_kernel[attract_blocks, threads_per_block](
            d_src,
            d_dst,
            d_weights,
            d_embeddings,
            learning_rate,
            max_loss,
            d_loss,
        )

        # Get loss for monitoring
        loss = d_loss.copy_to_host().sum() / n_edges
        epoch_range.set_description(f"[Loss: {loss:.4f}]")

    return d_embeddings.copy_to_host()


if __name__ == "__main__":
    dim = int(os.environ["DIM"])

    # Build kwargs for ggvec parameters from environment variables
    ggvec_kwargs = {}
    if "LEARNING_RATE" in os.environ:
        ggvec_kwargs["learning_rate"] = float(os.environ["LEARNING_RATE"])
    if "NEGATIVE_RATIO" in os.environ:
        ggvec_kwargs["negative_ratio"] = float(os.environ["NEGATIVE_RATIO"])
    if "NEGATIVE_DECAY" in os.environ:
        ggvec_kwargs["negative_decay"] = float(os.environ["NEGATIVE_DECAY"])
    if "MAX_LOSS" in os.environ:
        ggvec_kwargs["max_loss"] = float(os.environ["MAX_LOSS"])
    if "MAX_EPOCH" in os.environ:
        ggvec_kwargs["max_epoch"] = int(os.environ["MAX_EPOCH"])
    if "TOL" in os.environ:
        ggvec_kwargs["tol"] = float(os.environ["TOL"])
    if "TOL_SAMPLES" in os.environ:
        ggvec_kwargs["tol_samples"] = int(os.environ["TOL_SAMPLES"])

    if "N_THREADS" in os.environ:
        numba.set_num_threads(int(os.environ["N_THREADS"]))

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    ids = load(directory, "ids.npy")

    (sources, targets) = load_coo_array(directory, "mutual-edges-coo.vortex")

    # sources = load_array(directory, "edges-mutual-coo-sources.vortex")
    # targets = load_array(directory, "edges-mutual-coo-targets.vortex")
    weights = np.ones(len(sources), dtype=np.float32)

    G = scipy.sparse.coo_array(
        (weights, (sources, targets)), shape=(len(ids), len(ids))
    )

    embeddings = ggvec_cuda_main(G, n_components=dim, **ggvec_kwargs)

    save(directory, f"embeddings-{dim}-euclidean.npy", embeddings)
