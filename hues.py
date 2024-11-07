import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse.csgraph import shortest_path

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


def interpolate_point_hue(point, cluster_centers, cluster_hues, method='nearest_two'):
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
    # Calculate distances to all cluster centers
    distances = np.linalg.norm(cluster_centers - point, axis=1)

    if method == 'nearest_two':
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
        raise Exception("expected method to be 'nearest_two' or 'inverse_distance'")

    return interpolated_hue
