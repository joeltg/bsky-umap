import os
import sys
from typing import cast

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
# from umap import UMAP

from utils import load, save
from ums1 import project_embeddings

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

    embeddings: NDArray[np.float32] = load(directory, f"embeddings-{dim}.npy")
    knn_indices: NDArray[np.int32] = load(
        directory, f"knn_indices-{dim}-{metric}-{n_neighbors}.npy"
    )
    knn_dists: NDArray[np.float32] = load(
        directory, f"knn_dists-{dim}-{metric}-{n_neighbors}.npy"
    )

    # umap = UMAP(
    #     n_neighbors=n_neighbors,
    #     precomputed_knn=(knn_indices, knn_dists, None),
    #     spread=spread,
    #     min_dist=min_dist,
    #     n_epochs=n_epochs,
    #     n_jobs=n_threads,
    #     metric=metric,
    #     init="pca",
    #     verbose=True,
    # )

    # positions = cast(NDArray[np.float32], umap.fit_transform(embeddings))

    positions = project_embeddings(
        embeddings,
        n_neighbors=n_neighbors,
        knn=(knn_indices, knn_dists),
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        n_jobs=n_threads,
        metric=metric,
        init="pca",
    )

    save(directory, f"positions-{dim}-{metric}-{n_neighbors}.npy", positions)


if __name__ == "__main__":
    main()
