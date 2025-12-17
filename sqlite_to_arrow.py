import os
import sqlite3
import sys

import numpy as np
import pyarrow as pa
import tqdm
from dotenv import load_dotenv

from utils import save, write_edges, write_nodes

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

        rows = np.zeros(edge_count, dtype=np.int32)
        cols = np.zeros(edge_count, dtype=np.int32)

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
                rows[i] = s
                cols[i] = t
                outgoing_degrees[s] += 1
                incoming_degrees[t] += 1
                edge_rowid = rowid

    finally:
        conn.close()

    # Sort edges by source (primary) then target (secondary) for CSR representation
    print("Sorting edges...")
    edge_table = pa.table({"sources": rows, "targets": cols})

    sorted_table = edge_table.sort_by(
        [("sources", "ascending"), ("targets", "ascending")]
    )

    rows = sorted_table["sources"].to_numpy()
    cols = sorted_table["targets"].to_numpy()
    print("Edges sorted!")

    nodes_path = os.path.join(directory, "nodes.arrow")
    write_nodes(nodes_path, ids, incoming_degrees, outgoing_degrees)
    print("wrote", nodes_path)

    save(directory, "ids.npy", ids)
    save(directory, "sources.npy", rows)
    save(directory, "targets.npy", cols)
    save(directory, "incoming_degrees.npy", incoming_degrees)
    save(directory, "outgoing_degrees.npy", outgoing_degrees)

    print("computing weights")

    # normalize edge weights
    # w(u,v) = 64 * (
    #   min(ln(d_out(u)+1), ln(d_in(v)+1)) / max(ln(d_out(u)+1), ln(d_in(v)+1))
    # ) / ln((d_out(u)+1) * (d_in(v)+1))

    # Pre-compute log transforms once
    w_outgoing = np.log1p(outgoing_degrees, dtype=np.float32)
    w_incoming = np.log1p(incoming_degrees, dtype=np.float32)

    # Allocate final weights array
    weights = np.zeros(edge_count, dtype=np.float32)

    chunk_size = 1_000_000
    for start_idx in tqdm.trange(0, edge_count, chunk_size, desc="normalizing weights"):
        end_idx = min(start_idx + chunk_size, edge_count)

        # Get chunk of row/col indices
        chunk_rows = rows[start_idx:end_idx]
        chunk_cols = cols[start_idx:end_idx]

        # Compute weights for this chunk directly into the weights array
        w_src = w_outgoing[chunk_rows]
        w_dst = w_incoming[chunk_cols]
        w_min = np.minimum(w_src, w_dst)
        w_max = np.maximum(w_src, w_dst)

        weights[start_idx:end_idx] = 64.0 * (w_min / w_max) / (w_src + w_dst)

    print("done computing weights!")

    save(directory, "weights.npy", weights)

    edges_path = os.path.join(directory, "edges.arrow")
    write_edges(edges_path, (weights, rows, cols))

    ids_path = os.path.join(directory, "ids.buffer")
    ids.tofile(ids_path)
    print("wrote", ids_path)


if __name__ == "__main__":
    main()
