import sys
import os
import pickle

import numpy as np
from umap.umap_ import nearest_neighbors

n_neighbors = 5

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, 'graph-emb.pkl')
    neighbors_path = os.path.join(directory, 'graph-knn.pkl')

    with open(embedding_path, 'rb') as file:
        (names, embeddings) = pickle.load(file)

    knn = nearest_neighbors(
        embeddings,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds=None,
        angular=False,
        random_state=None,
        verbose=True
    )

    with open(neighbors_path, 'wb') as file:
        pickle.dump((names, knn), file)

if __name__ == "__main__":
    main()
