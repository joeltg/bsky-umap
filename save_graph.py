import os
import sqlite3
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from utils import NodeReader

load_dotenv()


def count(cursor: sqlite3.Cursor, table: str) -> int:
    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
    return cursor.fetchone()[0]


def main():
    n_neighbors = int(os.environ["N_NEIGHBORS"])
    dim = int(os.environ["DIM"])
    metric = os.environ["METRIC"]

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    with NodeReader(nodes_path) as reader:
        (node_ids, incoming_degrees) = reader.get_nodes()

    positions_path = os.path.join(
        directory, f"positions-{dim}-{metric}-{n_neighbors}.npy"
    )
    positions: NDArray[np.float32] = np.load(positions_path)
    print(
        "loaded positions",
        positions_path,
        positions.shape,
        positions.dtype,
    )

    database_path = os.path.join(directory, "positions.sqlite")

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")

        cursor.execute("DROP TABLE IF EXISTS nodes")
        cursor.execute(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY NOT NULL,
                x FLOAT NOT NULL,
                y FLOAT NOT NULL,
                mass FLOAT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                weight FLOAT NOT NULL
            )
            """
        )

        # Prepare the data for insertion
        scale = 1000
        nodes = [
            (int(id), float(p[0] * scale), float(p[1] * scale), 1)
            for id, p in zip(node_ids, positions, strict=False)
        ]

        # Insert the data into the table
        cursor.executemany(
            "INSERT INTO nodes (id, x, y, mass) VALUES (?, ?, ?, ?)", nodes
        )

        conn.commit()
        print("saved", database_path)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
