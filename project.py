import os
import sys
from typing import cast

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from umap import UMAP

load_dotenv()


def main():
    dim = int(os.environ["DIM"])
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    n_epochs = int(os.environ["N_EPOCHS"])
    n_threads = int(os.environ["N_THREADS"])
    spread = float(os.environ["SPREAD"])
    min_dist = float(os.environ["MIN_DIST"])
    metric = os.environ["METRIC"]

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    embeddings_path = os.path.join(directory, f"embeddings-{dim}.npy")
    embeddings: NDArray[np.float32] = np.load(embeddings_path)
    knn_indices_path = os.path.join(
        directory, f"knn_indices-{dim}-{metric}-{n_neighbors}.npy"
    )
    knn_indices: NDArray[np.uint32] = np.load(knn_indices_path)
    knn_dists_path = os.path.join(
        directory, f"knn_dists-{dim}-{metric}-{n_neighbors}.npy"
    )
    knn_dists: NDArray[np.float32] = np.load(knn_dists_path)

    umap = UMAP(
        n_neighbors=n_neighbors,
        precomputed_knn=(knn_indices, knn_dists, None),
        spread=spread,
        min_dist=min_dist,
        n_epochs=n_epochs,
        n_jobs=n_threads,
        # metric=metric,
        init="pca",
        verbose=True,
    )

    positions = cast(NDArray[np.float32], umap.fit_transform(embeddings))

    print(f"positions has shape {positions.shape} [{positions.dtype}]")

    positions_path = os.path.join(
        directory, f"positions-{dim}-{metric}-{n_neighbors}.npy"
    )

    print(f"saving {positions_path}")
    np.save(positions_path, positions)


if __name__ == "__main__":
    main()
