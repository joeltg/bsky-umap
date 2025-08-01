import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
from dotenv import load_dotenv
from hsluv import hsluv_to_rgb
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from utils import load, save

load_dotenv()


def derive_cluster_hues(
    cluster_centers: NDArray[np.float32], n_neighbors=15, n_cycles=1
) -> NDArray[np.float32]:
    """
    Derive hues (0-1) for clusters using MDS-style method on approximate neighbors.

    Parameters:
    cluster_centers: ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    n_neighbors: int
        Number of neighbors to consider for each cluster

    Returns:
    ndarray of shape (n_clusters,)
        Hue value (0-1) for each cluster
    """
    # Get number of clusters
    n_clusters = len(cluster_centers)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(cluster_centers)
    distances, indices = nbrs.kneighbors(cluster_centers)

    # Create adjacency matrix from nearest neighbors
    adjacency = np.zeros((n_clusters, n_clusters))
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip first neighbor (self)
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    # Convert adjacency to similarities
    # Use Gaussian kernel on distances
    similarities = np.zeros((n_clusters, n_clusters))
    sigma = np.median(distances[:, 1])  # Use median distance to first neighbor
    for i in range(n_clusters):
        dists = np.linalg.norm(cluster_centers - cluster_centers[i], axis=1)
        similarities[i] = np.exp(-(dists**2) / (2 * sigma**2))

    # Center the similarity matrix
    n = len(similarities)
    H = np.eye(n) - np.ones((n, n)) / n
    centered_similarities = H @ similarities @ H

    # Get the first eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(centered_similarities)
    hue_values = eigenvectors[:, -1]

    # Normalize to [0,1] range, then multiply by n_cycles
    hue_values = (hue_values - np.min(hue_values)) / (
        np.max(hue_values) - np.min(hue_values)
    )
    hue_values = (hue_values * n_cycles) % 1.0

    return hue_values


def interpolate_point_hue(
    point: NDArray[np.float32],
    cluster_centers: NDArray[np.float32],
    cluster_hues: NDArray[np.float32],
    n_cycles=1,
    n_neighbors=3,
):
    # Calculate distances to all cluster centers
    distances: NDArray[np.float32] = np.linalg.norm(cluster_centers - point, axis=1)

    # Get two nearest clusters
    closest_indices = np.argsort(distances)[:n_neighbors]
    dists = distances[closest_indices]

    if dists[0] == 0:
        return cluster_hues[closest_indices[0]]

    # Linear interpolation weight
    weights = np.array(dists)
    weights = 1 / (weights + 1e-8)
    weights = weights / weights.sum()

    # Get hues of closest clusters
    hues = cluster_hues[closest_indices]

    center_hue = hues[0]
    wrapped_hues = hues.copy()

    for i in range(1, n_neighbors):
        diff = wrapped_hues[i] - center_hue
        possible_diffs = [diff, diff + 1 / n_cycles, diff - 1 / n_cycles]
        best_diff = possible_diffs[np.argmin(np.abs(possible_diffs))]
        wrapped_hues[i] = center_hue + best_diff

    return (np.sum(wrapped_hues * weights)) % 1.0


def process_chunk(
    chunk_data: tuple[int, NDArray[np.float32], NDArray[np.float32]],
    cluster_centers: NDArray[np.float32],
    cluster_hues: NDArray[np.float32],
    n_cycles: int,
    n_neighbors: int,
) -> tuple[int, NDArray[np.int32]]:
    start_idx, points, node_mass = chunk_data

    assert len(points) == len(node_mass)
    chunk_colors = np.zeros(len(points), dtype=np.int32)

    saturation = 85

    for i, point in enumerate(points):
        hue = interpolate_point_hue(
            point=point,
            cluster_centers=cluster_centers,
            cluster_hues=cluster_hues,
            n_cycles=n_cycles,
            n_neighbors=n_neighbors,
        )

        rgb = hsluv_to_rgb((hue * 360, saturation, node_mass[i]))
        (r, g, b) = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        chunk_colors[i] = np.int32(b << 16 | g << 8 | r)

    return start_idx, chunk_colors


def main():
    dim = int(os.environ["DIM"])
    n_clusters = int(os.environ["N_CLUSTERS"])
    n_threads = int(os.environ["N_THREADS"])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    incoming_degrees: NDArray[np.uint32] = load(directory, "incoming_degrees.npy")
    embeddings: NDArray[np.float32] = load(directory, f"embeddings-{dim}.npy")
    cluster_centers: NDArray[np.float32] = load(
        directory, f"cluster_centers-{dim}-{n_clusters}.npy"
    )

    log_degrees = np.log1p(incoming_degrees) / np.log(10)
    log_degrees_norm = log_degrees / np.max(log_degrees)

    min_lightness = 40
    max_lightness = 90
    node_mass: NDArray[np.float32] = (
        min_lightness + (max_lightness - min_lightness) * log_degrees_norm
    )

    n_cycles = 5
    n_neighbors = 3

    cluster_hues = derive_cluster_hues(
        cluster_centers=cluster_centers, n_cycles=n_cycles
    )

    # Determine number of processes and chunk size
    n_samples = len(embeddings)
    chunk_size = n_samples // (n_threads * 4)
    print(f"Using {n_threads} processes with chunk size of {chunk_size}")

    # Prepare chunks
    chunks: list[tuple[int, NDArray[np.float32], NDArray[np.float32]]] = []
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunks.append((i, embeddings[i:end_idx], node_mass[i:end_idx]))

    # Prepare the partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk,
        cluster_centers=cluster_centers,
        cluster_hues=cluster_hues,
        n_cycles=n_cycles,
        n_neighbors=n_neighbors,
    )

    # Create the final array to store results
    colors = np.zeros(n_samples, dtype=np.int32)

    # Process chunks in parallel
    with Pool(processes=n_threads) as pool:
        for start_idx, chunk_colors in pool.imap_unordered(
            process_chunk_partial, chunks
        ):
            end_idx = min(start_idx + len(chunk_colors), n_samples)
            colors[start_idx:end_idx] = chunk_colors

    save(directory, "colors.npy", colors)


if __name__ == "__main__":
    main()
