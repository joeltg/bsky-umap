import sys
import os

import sqlite3
import numpy as np

from utils import write_nodes, write_edges

from dotenv import load_dotenv
load_dotenv()

def count(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM \"{table}\"")
    return cursor.fetchone()[0]

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph.sqlite')

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        node_count = count(cursor, "nodes")
        print(f"Number of nodes: {node_count}")

        ids = np.zeros(node_count, dtype=np.uint32)
        id_to_index: dict[int, int] = {}

        incoming_degrees = np.zeros(node_count, dtype=np.uint32)

        cursor.execute("SELECT id FROM nodes ORDER BY id")
        for i, (id,) in enumerate(cursor):
            ids[i] = id
            id_to_index[id] = i

        edge_count = count(cursor, "edges")
        print(f"Number of edges: {edge_count}")

        # Allocate weights
        weights = np.ones(edge_count, dtype=np.float32)
        rows = np.zeros(edge_count, dtype=np.uint32)
        cols = np.zeros(edge_count, dtype=np.uint32)

        # Fill indices array
        cursor.execute("SELECT source, target FROM edges")
        for i, (source, target) in enumerate(cursor):
            s = id_to_index[source]
            t = id_to_index[target]
            rows[i] = s
            cols[i] = t
            incoming_degrees[t] += 1

            if i > 0 and i % 10000000 == 0:
                progress = 100 * float(i / edge_count)
                print(f"loaded {i} edges out of {edge_count} ({progress:.2f}%)")

    finally:
        conn.close()

    nodes_path = os.path.join(directory, "nodes.arrow")
    write_nodes(nodes_path, ids, incoming_degrees)

    edges_path = os.path.join(directory, "edges.arrow")
    write_edges(edges_path, (weights, rows, cols))

if __name__ == "__main__":
    main()
