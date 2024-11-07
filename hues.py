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

def derive_cluster_hues(labels, cluster_centers):
    """
    Derive hues (0-1) for clusters using MDS-style method on cluster connectivity.

    Parameters:
    embeddings: ndarray of shape (n_samples, n_features)
        The input data points
    labels: ndarray of shape (n_samples,)
        Cluster labels for each point
    cluster_centers: ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers

    Returns:
    ndarray of shape (n_clusters,)
        Hue value (0-1) for each cluster
    """
    # Get number of clusters
    n_clusters = len(cluster_centers)

    # Find cluster neighbors using Voronoi tessellation
    vor = Voronoi(cluster_centers)

    # Create adjacency matrix from Voronoi ridge points
    adjacency = np.zeros((n_clusters, n_clusters))
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    # Get shortest path distances between all clusters
    distances = shortest_path(adjacency)

    # Convert distances to similarities
    max_dist = np.max(distances[distances != np.inf])
    similarities = 1 - (distances / max_dist)

    # Center the similarity matrix
    n = len(similarities)
    H = np.eye(n) - np.ones((n, n)) / n
    centered_similarities = H @ similarities @ H

    # Get the first eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(centered_similarities)
    hue_values = eigenvectors[:, -1]

    # Normalize to [0,1] range
    hue_values = (hue_values - np.min(hue_values)) / (np.max(hue_values) - np.min(hue_values))

    return hue_values


voronoi_cache = None

def interpolate_point_hue(point, cluster_centers, cluster_hues, method='voronoi_neighbors'):
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
        'nearest_two' or 'inverse_distance'

    Returns:
    float
        Interpolated hue value (0-1)
    """
    global voronoi_cache

    # Calculate distances to all cluster centers
    distances = np.linalg.norm(cluster_centers - point, axis=1)
    if method == 'voronoi_neighbors':
        # Create or use cached Voronoi diagram and neighbor dictionary
        if voronoi_cache is None:
            vor = Voronoi(cluster_centers)
            # Create dictionary of neighbors for each center
            neighbor_dict = {}
            for ridge_points in vor.ridge_points:
                i, j = ridge_points
                if i not in neighbor_dict:
                    neighbor_dict[i] = set()
                if j not in neighbor_dict:
                    neighbor_dict[j] = set()
                neighbor_dict[i].add(j)
                neighbor_dict[j].add(i)
            voronoi_cache = (vor, neighbor_dict)
        else:
            vor, neighbor_dict = voronoi_cache

        # Find nearest cluster center
        distances = np.linalg.norm(cluster_centers - point, axis=1)
        nearest_center_idx = np.argmin(distances)

        # Get its Voronoi neighbors
        neighbor_indices = list(neighbor_dict[nearest_center_idx])

        # Calculate distances to nearest center and its neighbors
        relevant_centers = [nearest_center_idx] + neighbor_indices
        relevant_distances = np.linalg.norm(
            cluster_centers[relevant_centers] - point[:, np.newaxis].T,
            axis=1
        )

        # Convert distances to weights using inverse distance
        weights = 1 / (relevant_distances + 1e-8)
        weights = weights / np.sum(weights)

        # Get weighted average of hues from neighboring centers
        relevant_hues = cluster_hues[relevant_centers]

        # Handle wraparound in hue space by finding the best wrapping
        # First, center all hues around the nearest center's hue
        center_hue = relevant_hues[0]
        wrapped_hues = relevant_hues.copy()
        for i in range(1, len(wrapped_hues)):
            diff = wrapped_hues[i] - center_hue
            if abs(diff) > 0.5:
                if diff > 0:
                    wrapped_hues[i] -= 1
                else:
                    wrapped_hues[i] += 1

        # Compute weighted average and wrap back to [0,1]
        interpolated_hue = (np.sum(wrapped_hues * weights)) % 1.0

        return interpolated_hue
    elif method == 'nearest_two':
        # Get two nearest clusters
        closest_indices = np.argsort(distances)[:2]
        d1, d2 = distances[closest_indices]

        # Edge case: point exactly on a cluster center
        if d1 == 0:
            return cluster_hues[closest_indices[0]]

        # Linear interpolation weight
        total_dist = d1 + d2
        weight = d1 / total_dist

        # Get hues of two closest clusters
        hue1 = cluster_hues[closest_indices[0]]
        hue2 = cluster_hues[closest_indices[1]]

        # Handle wraparound in hue space
        # If hues are more than 0.5 apart, wrap around the other way
        if abs(hue2 - hue1) > 0.5:
            if hue2 > hue1:
                hue2 -= 1
            else:
                hue2 += 1

        # Interpolate and wrap back to [0,1]
        interpolated_hue = (hue1 * (1 - weight) + hue2 * weight) % 1.0

    elif method == 'inverse_distance':
        # Prevent division by zero
        eps = 1e-8
        weights = 1 / (distances + eps)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average of hues
        # Note: this simple weighted average doesn't handle wraparound
        # as well as the two-point interpolation
        interpolated_hue = np.sum(cluster_hues * weights) % 1.0

    else:
        raise Exception("expected method to be 'voronoi_neighbors', 'nearest_two', or 'inverse_distance'")

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

    print("labels:", type(labels), labels.shape)
    print("cluster_centers:", type(cluster_centers), cluster_centers.shape)

    cluster_hues = derive_cluster_hues(labels=labels, cluster_centers=cluster_centers)

    hues = [interpolate_point_hue(point, cluster_centers, cluster_hues) for point in embeddings]

    save_labels(graph_database_path, node_ids, hues)
    save_labels(atlas_database_path, node_ids, hues)

if __name__ == "__main__":
    main()
