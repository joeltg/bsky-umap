import sys
import os
import pickle
import sqlite3

import numpy as np
from umap.umap_ import nearest_neighbors

from dotenv import load_dotenv

load_dotenv()


def main():
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    dim = int(os.environ['DIM'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    neighbors_path = os.path.join(directory, 'graph-knn-{:d}-{:d}.pkl'.format(dim, n_neighbors))
    # database_path = os.path.join(directory, 'graph-knn-' + str(n_neighbors) + '.sqlite')

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
        n_jobs=14,
    )

    with open(neighbors_path, 'wb') as file:
        pickle.dump((node_ids, knn), file)

    # conn = sqlite3.connect(database_path)
    # cursor = conn.cursor()

    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS nodes (
    #     id INTEGER PRIMARY KEY,
    #     x FLOAT NOT NULL DEFAULT 0,
    #     y FLOAT NOT NULL DEFAULT 0
    # )
    # ''')

    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS edges (
    #     source INTEGER NOT NULL,
    #     target INTEGER NOT NULL,
    #     weight FLOAT NOT NULL DEFAULT 1.0
    # );
    # ''')

    # cursor.execute("DELETE FROM nodes")
    # cursor.execute("DELETE FROM edges")

    # data = [(int(id),) for id in node_ids]
    # cursor.executemany("INSERT INTO nodes (id) VALUES (?)", data)

    # for i, id in enumerate(node_ids):
    #     data = [(int(id), int(node_ids[target]), float(dist)) for target, dist in zip(knn[0][i][1:], knn[1][i][1:])]
    #     cursor.executemany("INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)", data)

    # conn.commit()
    # conn.close()

if __name__ == "__main__":
    main()
