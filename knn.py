import sys
import os
import pickle

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
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    neighbors_path = os.path.join(directory, 'graph-knn-{:d}-{:d}.pkl'.format(dim, n_neighbors))

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    knn = nearest_neighbors(
        embeddings,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds=None,
        angular=False,
        random_state=None,
        verbose=True,
        n_jobs=n_threads,
    )

    with open(neighbors_path, 'wb') as file:
        pickle.dump((node_ids, knn), file)

if __name__ == "__main__":
    main()
