import os
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.cluster import KMeans

load_dotenv()


def main():
    dim = int(os.environ["DIM"])
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    n_clusters = int(os.environ["N_CLUSTERS"])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    high_embeddings_path = os.path.join(directory, f"high_embeddings-{dim}.npy")
    high_embeddings: NDArray[np.float32] = np.load(high_embeddings_path)

    print("high_embeddings", high_embeddings.shape)

    print("Performing k-means clustering")
    clusterer = KMeans(n_clusters=n_clusters, verbose=1, algorithm="elkan").fit(
        high_embeddings
    )
    print("k-means clustering completed.")

    cluster_labels = clusterer.labels_
    assert cluster_labels is not None
    print("labels", type(cluster_labels), cluster_labels.shape)

    cluster_centers = clusterer.cluster_centers_
    print("cluster_centers", type(cluster_centers), cluster_centers.shape)

    cluster_labels_path = os.path.join(
        directory, f"cluster_labels-{dim}-{n_neighbors}-{n_clusters}.npy"
    )
    np.save(cluster_labels_path, cluster_labels)

    cluster_centers_path = os.path.join(
        directory, f"cluster_centers-{dim}-{n_neighbors}-{n_clusters}.npy"
    )
    np.save(cluster_centers_path, cluster_centers)


if __name__ == "__main__":
    main()
