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

    ggvec_model = nodevectors.GGVec(
        n_components=dim,
        threads=14,
        verbose=True,
    )

    embeddings = ggvec_model.fit_transform(G)

    with open(embedding_path, 'wb') as file:
        pickle.dump((G.names, embeddings), file)

if __name__ == "__main__":
    main()
