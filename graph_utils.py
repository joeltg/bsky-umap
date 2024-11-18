import sqlite3
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import csrgraph as cg

edges_table='edges'
nodes_table='nodes'
source_col='source'
target_col='target'

def save_csr_graph(database_path, node_ids, csrgraph):
    coograph = csrgraph.tocoo()
    coograph.sum_duplicates()
    coograph.eliminate_zeros()

    if coograph.shape[0] != len(node_ids) or coograph.shape[1] != len(node_ids):
        raise Exception("unexpected matrix dimensions")

    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY,
        x FLOAT NOT NULL DEFAULT 0,
        y FLOAT NOT NULL DEFAULT 0,
        mass INTEGER NOT NULL DEFAULT 0,
        color INTEGER NOT NULL DEFAULT 0
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS edges (
        source INTEGER NOT NULL,
        target INTEGER NOT NULL
    );
    ''')

    try:
        conn.execute("BEGIN")
        cursor.execute("DELETE FROM nodes")
        cursor.execute("DELETE FROM edges")

        data = [(id) for id in zip(node_ids)]
        cursor.executemany("INSERT INTO nodes(rowid) VALUES (?)", data)

        data = [(int(node_ids[s]), int(node_ids[t]), float(w)) for s, t, w in zip(coograph.row, coograph.col, coograph.data)]
        cursor.executemany("INSERT INTO edges(source, target, weight) VALUES (?, ?, ?)", data)

        cursor.execute("CREATE INDEX IF NOT EXISTS edge_source ON edges(source)")
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"SQLite error: {e}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def load_csr_graph(database_path):
    (matrix, node_ids) = load_coo_matrix(database_path)
    return cg.csrgraph(matrix.tocsr(), node_ids, copy=False)

def load_coo_matrix(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT rowid FROM {nodes_table} ORDER BY rowid")
        node_ids = [row[0] for row in cursor.fetchall()]
        node_count = len(node_ids)
        print(f"Number of unique nodes: {node_count}")

        id_to_index = {id: index for index, id in enumerate(node_ids)}

        # Count edges
        cursor.execute(f"SELECT COUNT(*) FROM {edges_table}")
        edge_count = cursor.fetchone()[0]
        print(f"Number of edges: {edge_count}")

        # Allocate weights
        data = np.ones(edge_count, dtype=np.float64)

        rows = np.zeros(edge_count, dtype=np.int64)
        cols = np.zeros(edge_count, dtype=np.int64)

        # Fill indices array
        cursor.execute(f"SELECT {source_col}, {target_col} FROM {edges_table} ORDER BY {source_col}")
        for i, (source, target) in enumerate(cursor):
            rows[i] = id_to_index[source]
            cols[i] = id_to_index[target]

    finally:
        conn.close()

    matrix = coo_matrix((data, (rows, cols)), shape=(node_count, node_count));
    return (matrix, node_ids)

def save_colors(database_path, node_ids, colors):
    print("opening", database_path)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")
        data = [(int(color), id) for id, color in zip(node_ids, colors)]
        cursor.executemany("UPDATE nodes SET color = ? WHERE id = ?", data)
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"SQLite error: {e}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# def load_csr_matrix(database_path):
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()

#     try:
#         cursor.execute(f"SELECT rowid FROM {nodes_table} ORDER BY rowid")
#         node_ids = [row[0] for row in cursor.fetchall()]
#         node_count = len(node_ids)
#         print(f"Number of unique nodes: {node_count}")

#         id_to_index = {id: index for index, id in enumerate(node_ids)}

#         # Count edges
#         cursor.execute(f"SELECT COUNT(*) FROM {edges_table}")
#         edge_count = cursor.fetchone()[0]
#         print(f"Number of edges: {edge_count}")

#         # Preallocate arrays
#         indptr = np.zeros(node_count + 1, dtype=np.int64)
#         indices = np.empty(edge_count, dtype=np.int64)
#         data = np.ones(edge_count, dtype=np.float64)

#         # Count edges per source
#         cursor.execute(f"SELECT {source_col}, COUNT(*) FROM {edges_table} GROUP BY {source_col} ORDER BY {source_col}")
#         for source, count in cursor:
#             source_index = id_to_index[source]
#             indptr[source_index + 1] = count

#         # Cumulative sum to get correct indptr
#         np.cumsum(indptr, out=indptr)

#         # Fill indices array
#         cursor.execute(f"SELECT {source_col}, {target_col} FROM {edges_table} ORDER BY {source_col}")
#         for i, (source, target) in enumerate(cursor):
#             source_index = id_to_index[source]
#             target_index = id_to_index[target]
#             indices[i] = target_index

#     finally:
#         conn.close()

#     # Debug information
#     print(f"indptr shape: {indptr.shape}, last value: {indptr[-1]}")
#     print(f"indices shape: {indices.shape}")
#     print(f"data shape: {data.shape}")

#     # coo_matrix((data, (rows, cols)))

#     matrix = csr_matrix((data, indices, indptr), shape=(node_count, node_count));
#     return (matrix, node_ids)
