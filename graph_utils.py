import sqlite3
import numpy as np
from scipy.sparse import csr_matrix
import csrgraph as cg

def edges_to_csr_matrix(db_path, edges_table='edges', nodes_table='nodes', source_col='source', target_col='target'):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON')
    cursor = conn.cursor()

    try:
        conn.execute('BEGIN')

        # Get all unique node IDs
        cursor.execute(f"SELECT rowid FROM {nodes_table} ORDER BY rowid")
        node_ids = [row[0] for row in cursor.fetchall()]
        n = len(node_ids)
        print(f"Number of unique nodes: {n}")

        node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

        # Count edges
        cursor.execute(f"SELECT COUNT(*) FROM {edges_table}")
        nnz = cursor.fetchone()[0]
        print(f"Number of edges: {nnz}")

        # Preallocate arrays
        indptr = np.zeros(n + 1, dtype=np.int64)
        indices = np.empty(nnz, dtype=np.int64)
        data = np.ones(nnz, dtype=np.float64)

        # Count edges per source
        cursor.execute(f"SELECT {source_col}, COUNT(*) FROM {edges_table} GROUP BY {source_col} ORDER BY {source_col}")
        for source, count in cursor:
            source_index = node_to_index[source]
            indptr[source_index + 1] = count

        # Cumulative sum to get correct indptr
        np.cumsum(indptr, out=indptr)

        # Fill indices array
        cursor.execute(f"SELECT {source_col}, {target_col} FROM {edges_table} ORDER BY {source_col}")
        for i, (source, target) in enumerate(cursor):
            source_index = node_to_index[source]
            target_index = node_to_index[target]
            indices[i] = target_index

        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"SQLite error: {e}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    # Debug information
    print(f"indptr shape: {indptr.shape}, last value: {indptr[-1]}")
    print(f"indices shape: {indices.shape}")
    print(f"data shape: {data.shape}")

    csr_graph = csr_matrix((data, indices, indptr), shape=(n, n));

    return cg.csrgraph(csr_graph, node_ids)

# Example usage
if __name__ == "__main__":
    db_path = 'your_database.db'
    try:
        matrix = edges_to_csr_matrix(db_path)
        print(f"CSR Matrix shape: {matrix.shape}")
        print(f"Number of non-zero elements: {matrix.nnz}")
    except Exception as e:
        print(f"Error: {e}")
