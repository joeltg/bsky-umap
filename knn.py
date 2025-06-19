import sys
import os

import numpy as np
from numpy.typing import NDArray
from umap.umap_ import nearest_neighbors

from dotenv import load_dotenv

load_dotenv()

def main():
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    dim = int(os.environ['DIM'])
    n_threads = int(os.environ['N_THREADS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    high_embeddings_path = os.path.join(directory, f"high_embeddings-{dim}.npy")
    knn_indices_path = os.path.join(directory, f"knn_indices-{dim}-{n_neighbors}.npy")
    knn_dists_path = os.path.join(directory, f"knn_dists-{dim}-{n_neighbors}.npy")

    high_embeddings: NDArray[np.float32] = np.load(high_embeddings_path)

    print("loaded embeddings", high_embeddings.shape)

    knn = nearest_neighbors(
        high_embeddings,
        n_neighbors=n_neighbors,
        metric="cosine",
        metric_kwds=None,
        angular=True,
        random_state=None,
        verbose=True,
        n_jobs=n_threads,
    )
    print("finished nearest neighbors descent!")
    (knn_indices, knn_dists, rp_forest) = knn

    print("saving", knn_indices_path)
    np.save(knn_indices_path, knn_indices)
    print("saving", knn_dists_path)
    np.save(knn_dists_path, knn_dists)

if __name__ == "__main__":
    main()
