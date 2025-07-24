import os
import sys
from typing import cast

import numba
import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from umap.umap_ import simplicial_set_embedding

from utils import load, save

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
    # knn_indices: NDArray[np.int32] = load(
    #     directory, f"knn_indices-{dim}-{metric}-{n_neighbors}.npy"
    # )
    # knn_dists: NDArray[np.float32] = load(
    #     directory, f"knn_dists-{dim}-{metric}-{n_neighbors}.npy"
    # )

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

    # positions = project_embeddings(
    #     embeddings,
    #     n_neighbors=n_neighbors,
    #     knn=(knn_indices, knn_dists),
    #     min_dist=min_dist,
    #     spread=spread,
    #     n_epochs=n_epochs,
    #     n_jobs=n_threads,
    #     metric=metric,
    #     init="pca",
    # )

    rows = load(directory, f"fss_rows-{dim}-{metric}-{n_neighbors}.npy")
    cols = load(directory, f"fss_cols-{dim}-{metric}-{n_neighbors}.npy")
    vals = load(directory, f"fss_vals-{dim}-{metric}-{n_neighbors}.npy")

    size = embeddings.shape[0]
    graph = coo_matrix((vals, (rows, cols)), shape=(size, size), copy=False)

    random_state = np.random.RandomState()
    learning_rate: float=1.0
    negative_sample_rate: int = 5
    repulsion_strength: float = 1.0
    a, b = find_ab_params(spread, min_dist)
    numba.set_num_threads(n_threads)
    (positions, aux_data) = simplicial_set_embedding(
        embeddings,
        graph=graph,
        n_components=2,
        initial_alpha=learning_rate,
        a=a,
        b=b,
        gamma=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init="pca",
        random_state=random_state,
        metric=metric,
        metric_kwds={},
        densmap=False,
        densmap_kwds=None,
        output_dens=False,
        parallel=True,
        verbose=True,
    )

    save(directory, f"positions-{dim}-{metric}-{n_neighbors}.npy", positions)


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


if __name__ == "__main__":
    main()
