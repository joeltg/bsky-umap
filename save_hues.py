import os
import sys
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial import Voronoi
from scipy.sparse.csgraph import shortest_path
from scipy import sparse
from scipy import stats
from sklearn.neighbors import NearestNeighbors

from dotenv import load_dotenv
load_dotenv()

from graph_utils import save_colors

def derive_cluster_hues(labels, cluster_centers, n_neighbors=15, n_cycles=1):
    """
    Derive hues (0-1) for clusters using MDS-style method on approximate neighbors.

    Parameters:
    labels: ndarray of shape (n_samples,)
        Cluster labels for each point
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

    # Use NearestNeighbors instead of Voronoi
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
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
        similarities[i] = np.exp(-dists**2 / (2 * sigma**2))

    # Center the similarity matrix
    n = len(similarities)
    H = np.eye(n) - np.ones((n, n)) / n
    centered_similarities = H @ similarities @ H

    # Get the first eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(centered_similarities)
    hue_values = eigenvectors[:, -1]

    # Normalize to [0,1] range, then multiply by n_cycles
    hue_values = (hue_values - np.min(hue_values)) / (np.max(hue_values) - np.min(hue_values))
    hue_values = (hue_values * n_cycles) % 1.0

    return hue_values

def interpolate_point_hue(point, cluster_centers, cluster_hues, n_cycles=1, n_neighbors=3):
    # Calculate distances to all cluster centers
    distances = np.linalg.norm(cluster_centers - point, axis=1)

    # Get two nearest clusters
    closest_indices = np.argsort(distances)[:n_neighbors]
    dists = distances[closest_indices]

    if dists[0] == 0:
        return cluster_hues[closest_indices[0]]

    # Linear interpolation weight
    weights = np.array(dists)
    total_dist = weights.sum()
    weights = 1 / (weights + 1e-8)
    weights = weights / weights.sum()

    # Get hues of closest clusters
    hues = cluster_hues[closest_indices]

    center_hue = hues[0]
    wrapped_hues = hues.copy()

    for i in range(1, n_neighbors):
        diff = wrapped_hues[i] - center_hue
        possible_diffs = [
            diff,
            diff + 1/n_cycles,
            diff - 1/n_cycles
        ]
        best_diff = possible_diffs[np.argmin(np.abs(possible_diffs))]
        wrapped_hues[i] = center_hue + best_diff

    return (np.sum(wrapped_hues * weights)) % 1.0

def process_chunk(chunk_data, cluster_centers, cluster_hues, n_cycles, n_neighbors):
    start_idx, points = chunk_data
    chunk_hues = np.zeros(len(points), dtype=np.float32)

    for i, point in enumerate(points):
        chunk_hues[i] = interpolate_point_hue(
            point=point,
            cluster_centers=cluster_centers,
            cluster_hues=cluster_hues,
            n_cycles=n_cycles,
            n_neighbors=n_neighbors
        )

    return start_idx, chunk_hues

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, f'graph-emb-{dim}.pkl')
    label_path = os.path.join(directory, f'graph-label-{dim}-{n_neighbors}.pkl')
    graph_database_path = os.path.join(directory, f'graph-umap-{dim}-{n_neighbors}.sqlite')
    atlas_database_path = os.path.join(directory, f'atlas-umap-{dim}-{n_neighbors}.sqlite')

    print("embedding_path", embedding_path)
    print("label_path", label_path)
    print("graph_database_path", graph_database_path)
    print("atlas_database_path", atlas_database_path)

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    with open(label_path, 'rb') as file:
        (node_ids, labels, cluster_centers) = pickle.load(file)

    n_cycles = 5
    n_neighbors = 3

    cluster_hues = derive_cluster_hues(
        labels=labels,
        cluster_centers=cluster_centers,
        n_cycles=n_cycles
    )

    # Determine number of processes and chunk size
    n_processes = max(cpu_count() - 1, 1)
    n_samples = len(embeddings)
    chunk_size = n_samples // (n_processes * 4)
    print(f"Using {n_processes} processes with chunk size of {chunk_size}")

    # Prepare chunks
    chunks = []
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunks.append((i, embeddings[i:end_idx]))

    # Prepare the partial function with fixed arguments
    process_chunk_partial = partial(
        process_chunk,
        cluster_centers=cluster_centers,
        cluster_hues=cluster_hues,
        n_cycles=n_cycles,
        n_neighbors=n_neighbors
    )

    # Create the final array to store results
    hues = np.zeros(n_samples, dtype=np.int32)

    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        total_chunks = len(chunks)
        for chunk_num, (start_idx, chunk_hues) in enumerate(pool.imap_unordered(process_chunk_partial, chunks)):
            end_idx = min(start_idx + len(chunk_hues), n_samples)
            hues[start_idx:end_idx] = (chunk_hues * 256).astype(np.int32)

            if (chunk_num + 1) % max(1, total_chunks // 20) == 0:
                print(f"Processed {chunk_num + 1}/{total_chunks} chunks ({((chunk_num + 1) / total_chunks) * 100:.1f}%)")

    save_colors(graph_database_path, node_ids, hues)
    save_colors(atlas_database_path, node_ids, hues)

if __name__ == "__main__":
    main()
