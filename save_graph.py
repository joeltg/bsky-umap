import os
import sqlite3
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from utils import load

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

    ids: NDArray[np.uint32] = load(directory, "ids.npy")
    colors: NDArray[np.uint32] = load(directory, "colors.npy")
    positions: NDArray[np.float32] = load(
        directory, f"positions-{dim}-{metric}-{n_neighbors}.npy"
    )

    cols: NDArray[np.int32] = load(
        directory, f"fss_cols-{dim}-{metric}-{n_neighbors}.npy"
    )
    rows: NDArray[np.int32] = load(
        directory, f"fss_rows-{dim}-{metric}-{n_neighbors}.npy"
    )
    vals: NDArray[np.float32] = load(
        directory, f"fss_vals-{dim}-{metric}-{n_neighbors}.npy"
    )

    database_path = os.path.join(directory, "atlas.sqlite")

    conn = sqlite3.connect(database_path)

    try:
        conn.execute("BEGIN")

        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS nodes")
        cursor.execute(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY NOT NULL,
                x FLOAT NOT NULL,
                y FLOAT NOT NULL,
                mass FLOAT NOT NULL,
                color INTEGER NOT NULL
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
        cursor.executemany(
            "INSERT INTO nodes (id, x, y, mass, color) VALUES (?, ?, ?, ?, ?)",
            [
                (int(id), float(p[0]), float(p[1]), 1, int(c))
                for id, p, c in zip(ids, positions * scale, colors, strict=False)
            ],
        )

        cursor.executemany(
            "INSERT INTO edges (source, target, weight) VALUES (?, ?, ?)",
            [
                (int(ids[row]), int(ids[col]), float(val))
                for row, col, val in zip(rows, cols, vals, strict=False)
            ],
        )

        conn.commit()
        print("saved", database_path)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
