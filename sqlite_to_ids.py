import os
import sqlite3
import sys

import numpy as np
from dotenv import load_dotenv

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

        cursor.execute("SELECT id FROM nodes ORDER BY id")
        for i, (id,) in enumerate(cursor):
            ids[i] = id

    finally:
        conn.close()

    ids_path = os.path.join(directory, "ids.buffer")
    ids.tofile(ids_path)
    print("wrote", ids_path)


if __name__ == "__main__":
    main()
