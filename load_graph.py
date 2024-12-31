import sys
import os

import pickle

from dotenv import load_dotenv
load_dotenv()

from graph_utils import load_coo_matrix

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph.sqlite')
    matrix_path = os.path.join(directory, "graph-coo-matrix.pkl")

    (coo_matrix, node_ids) = load_coo_matrix(database_path)
    with open(matrix_path, 'wb') as file:
        pickle.dump((coo_matrix, node_ids), file)

if __name__ == "__main__":
    main()
