import os
import sys
import pickle
import sqlite3
import numpy as np

from umap import UMAP

from dotenv import load_dotenv

load_dotenv()

from graph_utils import save_csr_graph

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    n_epochs = int(os.environ['N_EPOCHS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    neighbors_path = os.path.join(directory, 'graph-knn-{:d}-{:d}.pkl'.format(dim, n_neighbors))

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS nodes")
    cursor.execute('''
    CREATE TABLE nodes (
        id INTEGER PRIMARY KEY NOT NULL,
        x FLOAT NOT NULL DEFAULT 0,
        y FLOAT NOT NULL DEFAULT 0,
        mass FLOAT NOT NULL DEFAULT 0,
        color FLOAT NOT NULL DEFAULT 0
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS edges (
        source INTEGER NOT NULL,
        target INTEGER NOT NULL
    );
    ''')

    with open(embedding_path, 'rb') as file:
        (node_ids, high_embeddings) = pickle.load(file)

    with open(neighbors_path, 'rb') as file:
        (_, knn) = pickle.load(file)

    n_neighbors = knn[0].shape[1]

    low_embeddings = UMAP(
        # n_components=3,
        n_neighbors=n_neighbors,
        precomputed_knn=knn,
        spread=4,
        min_dist=0.5,
        n_epochs=n_epochs,
        n_jobs=14,
        verbose=True
    ).fit_transform(high_embeddings)

    print("result.shape", low_embeddings.shape, type(low_embeddings))
    print("node_ids", node_ids.shape)

    # Prepare the data for insertion
    scale = 20000
    data = [(int(id), float(p[0] * scale), float(p[1] * scale)) for id, p in zip(node_ids, low_embeddings)]

    # Insert the data into the table
    cursor.executemany("INSERT INTO nodes (id, x, y) VALUES (?, ?, ?)", data)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
