import sys
import os

import nodevectors
import pickle
import csrgraph as cg

from dotenv import load_dotenv
load_dotenv()

def main():
    dim = int(os.environ['DIM'])
    n_threads = int(os.environ['N_THREADS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    matrix_path = os.path.join(directory, "graph-coo-matrix.pkl")
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))

    with open(matrix_path, 'rb') as file:
        (coo_matrix, node_ids) = pickle.load(file)

    G = cg.csrgraph(coo_matrix.tocsr(), node_ids, copy=False)

    embeddings = nodevectors.GGVec(
        n_components=dim,
        threads=n_threads,
        verbose=True,
    ).fit_transform(G)

    # embeddings = nodevectors.ProNE(
    #     n_components=dim,
    #     verbose=True
    # ).fit_transform(G)

    with open(embedding_path, 'wb') as file:
        pickle.dump((G.names, embeddings), file)

if __name__ == "__main__":
    main()
