import os
import sys
import numpy as np

from numpy.typing import NDArray
from umap import UMAP


from dotenv import load_dotenv

from graph_utils import write_edges

load_dotenv()

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    n_epochs = int(os.environ['N_EPOCHS'])
    n_threads = int(os.environ['N_THREADS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    high_embeddings_path = os.path.join(directory, f"high_embeddings-{dim}.npy")
    high_embeddings: NDArray[np.float32] = np.load(high_embeddings_path)
    knn_indices_path = os.path.join(directory, f"knn_indices-{dim}-{n_neighbors}.npy")
    knn_indices: NDArray[np.uint32] = np.load(knn_indices_path)
    knn_dists_path = os.path.join(directory, f"knn_dists-{dim}-{n_neighbors}.npy")
    knn_dists: NDArray[np.float32] = np.load(knn_dists_path)

    umap = UMAP(
        n_neighbors=n_neighbors,
        precomputed_knn=(knn_indices, knn_dists, None),
        spread=4,
        min_dist=0.5,
        n_epochs=n_epochs,
        n_jobs=n_threads,
        verbose=True
    )

    # (low_embeddings, graph, sigmas, rhos) = umap.fit_transform(high_embeddings)
    low_embeddings = umap.fit_transform(high_embeddings)

    print("low_embeddings has shape", low_embeddings.shape)

    low_embeddings_path = os.path.join(directory, f"low_embeddings-{dim}-{n_neighbors}.npy")
    print("saving", low_embeddings_path)
    np.save(low_embeddings_path, low_embeddings)

    # knn_edges_path = os.path.join(directory, f"knn_edges-{dim}-{n_neighbors}.arrow")
    # print("saving", knn_edges_path)
    # write_edges(knn_edges_path, (graph.data, graph.row, graph.col))

if __name__ == "__main__":
    main()
