import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import sqlite3

# Constants
N_NODES = 100_000
N_COMPONENTS = 1000
DATABASE = '/Users/joelgustafson/Downloads/graph-100000.sqlite'

def load_matrix(cursor, shape, batch_size=10000):
    # Create an empty LIL matrix
    matrix = sparse.lil_matrix(shape)

    # Fetch and add edges in batches
    offset = 0
    while True:
        cursor.execute("""
            SELECT source, target
            FROM edges
            LIMIT ? OFFSET ?
        """, (batch_size, offset))

        batch = cursor.fetchall()
        if not batch:
            break

        # print("got batch", batch)

        # Add edges to the matrix
        for source, target in batch:
            matrix[source-1, target-1] = 1

        offset += batch_size
        print(f"Processed {offset} edges")

    # Convert to CSR format for efficient computations
    return matrix.tocsr()

def main():
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()


    print("Loading data...")
    adj_matrix = load_matrix(cursor, shape=(N_NODES,N_NODES))

    # edges = load_data(cursor)
    # conn.close()

    # print("Creating sparse matrix...")
    # adj_matrix = create_sparse_matrix(edges, (N_NODES, N_NODES))

    print("Sparse matrix created. Shape:", adj_matrix.shape)
    print("Number of non-zero elements:", adj_matrix.nnz)
    print("Sparsity: {:.6f}%".format(100 * adj_matrix.nnz / (N_NODES * N_NODES)))

    n_components = 100

    print("Performing Truncated SVD (equivalent to PCA for this case)...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    transformed_data = svd.fit_transform(adj_matrix)

    print("SVD completed.")
    print("Transformed data shape:", transformed_data.shape)
    # print("Transformed data:", transformed_data)

    # # The right singular vectors are equivalent to the PCA components
    # components = svd.components_

    # # Optionally, you can save the results
    # np.save('pca_components.npy', components)
    # np.save('transformed_data.npy', transformed_data)

    # print("Results saved. Process complete.")

if __name__ == "__main__":
    main()
