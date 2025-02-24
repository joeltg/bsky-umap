import sys
import os

import sqlite3
import numpy as np
from numpy.typing import NDArray

from utils import read_nodes, read_edges

from dotenv import load_dotenv
load_dotenv()

def count(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM \"{table}\"")
    return cursor.fetchone()[0]

def main():
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    dim = int(os.environ['DIM'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    (ids, incoming_degrees) = read_nodes(nodes_path)

    low_embeddings_path = os.path.join(directory, f"low_embeddings-{dim}-{n_neighbors}.npy")
    low_embeddings: NDArray[np.float32] = np.load(low_embeddings_path)
    print("loaded low_embeddings", low_embeddings_path, low_embeddings.shape)

    # knn_edges_path = os.path.join(directory, f"knn_edges-{dim}-{n_neighbors}.arrow")
    # (weights, sources, targets) = read_edges(knn_edges_path, ids)
    # print("loaded knn_edges", knn_edges_path)

    database_path = os.path.join(directory, 'positions.sqlite')

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")
        cursor.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY NOT NULL, x FLOAT NOT NULL, y FLOAT NOT NULL, mass FLOAT NOT NULL)")
        cursor.execute("CREATE TABLE edges (source INTEGER NOT NULL, target INTEGER NOT NULL, weight FLOAT NOT NULL)")

        # Prepare the data for insertion
        scale = 1000
        # nodes = [(int(id), float(p[0] * scale), float(p[1] * scale), float(d + 1)) for id, d, p in zip(ids, incoming_degrees, low_embeddings)]
        nodes = [(int(id), float(p[0] * scale), float(p[1] * scale), 1) for id, p in zip(ids, low_embeddings)]

        # Insert the data into the table
        cursor.executemany("INSERT INTO nodes (id, x, y, mass) VALUES (?, ?, ?, ?)", nodes)

        # edges = [(int(ids[source]), int(ids[target]), float(weight)) for (weight, source, target) in zip(weights, sources, targets)]
        # cursor.executemany("INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)", edges)

        conn.commit()
        print("saved", database_path)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
