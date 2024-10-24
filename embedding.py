import sys
import os

import csrgraph as cg
import nodevectors
import numpy as np
import pickle

from graph_utils import load_csr_graph

dim = 32

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    database_path = os.path.join(directory, 'graph.sqlite')
    embedding_path = os.path.join(directory, 'graph-emb.pkl')

    G = load_csr_graph(database_path)
    ggvec_model = nodevectors.GGVec(n_components=dim, verbose=True)
    embeddings = ggvec_model.fit_transform(G)

    with open(embedding_path, 'wb') as file:
        pickle.dump((G.names, embeddings), file)

if __name__ == "__main__":
    main()
