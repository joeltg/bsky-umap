import os
import sqlite3
import sys

import numpy as np
import tqdm
from dotenv import load_dotenv

from utils import save

load_dotenv()


def count(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
    return cursor.fetchone()[0]


def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, "graph.sqlite")

    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        node_count = count(cursor, "nodes")
        print(f"Number of nodes: {node_count}")

        ids = np.zeros(node_count, dtype=np.uint32)
        id_to_index: dict[int, int] = {}

        incoming_degrees = np.zeros(node_count, dtype=np.uint32)
        outgoing_degrees = np.zeros(node_count, dtype=np.uint32)

        cursor.execute("SELECT id FROM nodes ORDER BY id")
        for i, (id,) in enumerate(cursor):
            ids[i] = id
            id_to_index[id] = i

        edge_count = count(cursor, "edges")
        print(f"Number of edges: {edge_count}")

        sources = np.zeros(edge_count, dtype=np.uint32)
        targets = np.zeros(edge_count, dtype=np.uint32)

        chunk_size = 1_000_000
        edge_rowid = 0
        for start_idx in tqdm.trange(0, edge_count, chunk_size, desc="loading edges"):
            cursor.execute(
                "SELECT rowid, source, target FROM edges WHERE rowid > ? LIMIT ?",
                (edge_rowid, chunk_size),
            )

            for offset, (rowid, source, target) in enumerate(cursor):
                i = start_idx + offset
                s = id_to_index[source]
                t = id_to_index[target]
                sources[i] = s
                targets[i] = t
                outgoing_degrees[s] += 1
                incoming_degrees[t] += 1
                edge_rowid = rowid

    finally:
        conn.close()

    save(directory, "ids.npy", ids)
    save(directory, "sources.npy", sources)
    save(directory, "targets.npy", targets)
    save(directory, "incoming_degrees.npy", incoming_degrees)
    save(directory, "outgoing_degrees.npy", outgoing_degrees)


if __name__ == "__main__":
    main()
