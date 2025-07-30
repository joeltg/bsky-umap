import os
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from utils import load, save

load_dotenv()


def main():
    dim = int(os.environ["DIM"])
    n_clusters = int(os.environ["N_CLUSTERS"])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    embeddings: NDArray[np.float32] = load(directory, f"embeddings-{dim}.npy")

    print("Performing k-means clustering")
    clusterer = KMeans(n_clusters=n_clusters, verbose=1, algorithm="elkan").fit(
        embeddings
    )
    print("k-means clustering completed.")

    cluster_labels = clusterer.labels_
    cluster_centers = clusterer.cluster_centers_
    assert cluster_labels is not None

    save(
        directory,
        f"cluster_labels-{dim}-{n_clusters}.npy",
        cluster_labels.astype(np.int32),
    )

    save(
        directory,
        f"cluster_centers-{dim}-{n_clusters}.npy",
        cluster_centers.astype(np.float32),
    )


if __name__ == "__main__":
    main()
