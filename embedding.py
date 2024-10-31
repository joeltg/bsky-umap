import sys
import os

import csrgraph as cg
import nodevectors
import numpy as np
import pickle

from dotenv import load_dotenv
load_dotenv()

from graph_utils import load_csr_graph

def main():
    dim = int(os.environ['DIM'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph.sqlite')
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))

    G = load_csr_graph(database_path)

    embeddings = nodevectors.GGVec(
        n_components=dim,
        threads=14,
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
