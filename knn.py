import os
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from pynndescent import NNDescent

load_dotenv()


def main():
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    dim = int(os.environ["DIM"])
    n_threads = int(os.environ["N_THREADS"])
    metric = os.environ["METRIC"]

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embeddings_path = os.path.join(directory, f"embeddings-{dim}.npy")
    knn_indices_path = os.path.join(
        directory, f"knn_indices-{dim}-{metric}-{n_neighbors}.npy"
    )
    knn_dists_path = os.path.join(
        directory, f"knn_dists-{dim}-{metric}-{n_neighbors}.npy"
    )

    embeddings: NDArray[np.float32] = np.load(embeddings_path)

    print(f"loaded embeddings {embeddings.shape} [{embeddings.dtype}]")

    knn_search_index = NNDescent(
        data=embeddings,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=None,
        random_state=None,
        low_memory=True,
        n_jobs=n_threads,
        verbose=True,
        compressed=False,
    )

    assert knn_search_index.neighbor_graph is not None
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    print("finished nearest neighbors descent!")

    print(f"saving {knn_indices_path} {knn_indices.shape} [{knn_indices.dtype}]")
    np.save(knn_indices_path, knn_indices)

    print(f"saving {knn_dists_path} {knn_dists.shape} [{knn_dists.dtype}]")
    np.save(knn_dists_path, knn_dists)


if __name__ == "__main__":
    main()
