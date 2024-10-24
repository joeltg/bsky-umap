import os
import sys
import pickle
import sqlite3
import numpy as np

from umap import UMAP

from graph_utils import save_csr_graph

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph.sqlite')
    embedding_path = os.path.join(directory, 'graph-emb.pkl')
    neighbors_path = os.path.join(directory, 'graph-knn.pkl')

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        x FLOAT NOT NULL DEFAULT 0,
        y FLOAT NOT NULL DEFAULT 0
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

    low_embeddings, knn_graph = UMAP(
        n_neighbors=n_neighbors,
        precomputed_knn=knn,
        spread=5.0,
        min_dist=2.0,
        # init="pca",
        # n_epochs=0,
        verbose=False
    ).fit_transform(high_embeddings)

    print("result.shape", low_embeddings.shape, type(low_embeddings))
    print("node_ids", node_ids.shape)

    save_csr_graph(os.path.join(directory, 'graph-knn.sqlite'), node_ids, knn_graph)

    # Prepare the data for insertion
    scale = 500
    data = [(int(id), float(p[0] * scale), float(p[1] * scale)) for id, p in zip(node_ids, low_embeddings)]

    # Insert the data into the table
    cursor.executemany('''
    INSERT INTO nodes (rowid, x, y) VALUES (?, ?, ?)
        ON CONFLICT(rowid) DO UPDATE SET x = excluded.x, y = excluded.y
    ''', data)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
