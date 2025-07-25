import os
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from pynndescent import NNDescent

from utils import load, save

load_dotenv()


def main():
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    dim = int(os.environ["DIM"])
    n_threads = int(os.environ["N_THREADS"])

    n_trees: None | int = None
    if "N_TREES" in os.environ:
        n_trees = int(os.environ["N_TREES"])

    n_iters: None | int = None
    if "N_ITERS" in os.environ:
        n_iters = int(os.environ["N_ITERS"])

    metric: str = "euclidean"
    if "METRIC" in os.environ:
        metric = os.environ["METRIC"]

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    embeddings: NDArray[np.float32] = load(directory, f"embeddings-{dim}.npy").copy()

    knn_search_index = NNDescent(
        data=embeddings,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=None,
        random_state=None,
        low_memory=True,
        n_jobs=n_threads,
        n_trees=n_trees,
        n_iters=n_iters,
        compressed=False,
        verbose=True,
    )

    assert knn_search_index.neighbor_graph is not None
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    print("finished nearest neighbors descent!")

    save(
        directory,
        f"knn_indices-{dim}-{metric}-{n_neighbors}.npy",
        knn_indices.astype(np.int32),
    )

    save(
        directory,
        f"knn_dists-{dim}-{metric}-{n_neighbors}.npy",
        knn_dists.astype(np.float32),
    )


if __name__ == "__main__":
    main()
