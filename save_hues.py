import os
import sys
import pickle
import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse.csgraph import shortest_path
from scipy import sparse
from scipy import stats

from dotenv import load_dotenv
load_dotenv()

from graph_utils import save_labels

import numpy as np
from sklearn.neighbors import NearestNeighbors

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

    return hue_values, nbrs

def interpolate_point_hue(point, cluster_centers, cluster_hues, method='nearest_two', nbrs=None, n_cycles=1):
    """
    Interpolate a hue value for a point based on nearby cluster hues.

    Parameters:
    point: ndarray of shape (n_features,)
        Coordinates of the point
    cluster_centers: ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    cluster_hues: ndarray of shape (n_clusters,)
        Base hue value (0-1) for each cluster
    method: str
        'nearest_two', 'inverse_distance', or 'neighbors'
    nbrs: NearestNeighbors, optional
        Pre-fitted NearestNeighbors model for cluster centers

    Returns:
    float
        Interpolated hue value (0-1)
    """
    if method == 'neighbors':
        if nbrs is None:
            raise ValueError("neighbors method requires a fitted NearestNeighbors model")

        # Find nearest cluster center and its neighbors
        distances, indices = nbrs.kneighbors([point], n_neighbors=5)
        distances = distances[0]
        indices = indices[0]

        # Convert distances to weights using inverse distance
        weights = 1 / (distances + 1e-8)
        weights = weights / np.sum(weights)

        # Get weighted average of hues from neighboring centers
        relevant_hues = cluster_hues[indices]

        # Handle wraparound in hue space with n_cycles
        center_hue = relevant_hues[0]
        wrapped_hues = relevant_hues.copy()
        for i in range(1, len(wrapped_hues)):
            diff = wrapped_hues[i] - center_hue
            # Consider wrapping in either direction
            possible_diffs = [
                diff,  # no wrap
                diff + 1/n_cycles,  # wrap one way
                diff - 1/n_cycles   # wrap other way
            ]
            # Use the smallest absolute difference
            best_diff = possible_diffs[np.argmin(np.abs(possible_diffs))]
            wrapped_hues[i] = center_hue + best_diff

        interpolated_hue = (np.sum(wrapped_hues * weights)) % 1.0

        return interpolated_hue

    elif method == 'nearest_two':
        # Calculate distances to all cluster centers
        distances = np.linalg.norm(cluster_centers - point, axis=1)

        # Get two nearest clusters
        closest_indices = np.argsort(distances)[:2]
        d1, d2 = distances[closest_indices]

        if d1 == 0:
            return cluster_hues[closest_indices[0]]

        # Linear interpolation weight
        total_dist = d1 + d2
        weight = d1 / total_dist

        # Get hues of two closest clusters
        hue1 = cluster_hues[closest_indices[0]]
        hue2 = cluster_hues[closest_indices[1]]

        # Handle wraparound with n_cycles
        # Consider all possible paths between the hues
        diff = hue2 - hue1
        possible_diffs = [
            diff,  # no wrap
            diff + 1/n_cycles,  # wrap one way
            diff - 1/n_cycles   # wrap other way
        ]
        # Use the smallest absolute difference
        diff = possible_diffs[np.argmin(np.abs(possible_diffs))]

        interpolated_hue = (hue1 + diff * weight) % 1.0

        return interpolated_hue

    else:
        raise Exception("oh no")

    return interpolated_hue

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    label_path = os.path.join(directory, 'graph-label-{:d}-{:d}.pkl'.format(dim, n_neighbors))
    graph_database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))
    atlas_database_path = os.path.join(directory, 'atlas-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))

    print("embedding_path", embedding_path)
    print("label_path", label_path)
    print("graph_database_path", graph_database_path)
    print("atlas_database_path", atlas_database_path)

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("embeddings:", type(embeddings), embeddings.shape)

    with open(label_path, 'rb') as file:
        (node_ids, labels, cluster_centers) = pickle.load(file)

    print("labels:", type(labels), labels)
    print("cluster_centers:", type(cluster_centers), cluster_centers.shape)

    log_frequency = 10000
    method = 'nearest_two'
    n_cycles = 5

    (cluster_hues, nbrs) = derive_cluster_hues(
        labels=labels,
        cluster_centers=cluster_centers,
        n_cycles=n_cycles
    )

    print("cluster_hues:", type(cluster_hues))

    n_samples = len(embeddings)
    hues = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        hue = interpolate_point_hue(
            point=embeddings[i],
            cluster_centers=cluster_centers,
            cluster_hues=cluster_hues,
            method=method,
            nbrs=nbrs,
            n_cycles=n_cycles
        )

        hues[i] = int(hue * 256)

        if (i + 1) % log_frequency == 0:
            print(f"Processed {i + 1}/{n_samples} points ({((i + 1) / n_samples) * 100:.1f}%)")

    save_labels(graph_database_path, node_ids, hues)
    save_labels(atlas_database_path, node_ids, hues)

if __name__ == "__main__":
    main()
