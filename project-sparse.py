import os
import sys
import pickle
import sqlite3
import csrgraph as cg
import numpy as np
from umap import UMAP

from dotenv import load_dotenv

load_dotenv()

from graph_utils import load_coo_matrix

def main():
    n_epochs = int(os.environ['N_EPOCHS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    input_database_path = os.path.join(directory, 'graph.sqlite')
    output_database_path = os.path.join(directory, 'graph-umap-sparse.pkl')

    (matrix, node_ids) = load_coo_matrix(input_database_path)
    low_embeddings = UMAP(
        n_neighbors=n_neighbors,
        spread=4,
        min_dist=0.5,
        n_epochs=n_epochs,
        n_jobs=14,
        verbose=True
    ).fit_transform(matrix.tocsr())

    with open(output_path, 'wb') as file:
        pickle.dump((node_ids, low_embeddings), file)

if __name__ == "__main__":
    main()
