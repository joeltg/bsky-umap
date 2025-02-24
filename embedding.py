import sys
import os
import nodevectors

import numpy as np
import csrgraph as cg
from scipy.sparse import coo_matrix

from graph_utils import read_nodes, read_edges

from dotenv import load_dotenv
load_dotenv()

def main():
    dim = int(os.environ['DIM'])
    n_threads = int(os.environ['N_THREADS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    (node_ids, incoming_degrees) = read_nodes(nodes_path)

    edges_path = os.path.join(directory, "edges.arrow")
    (weights, sources, targets) = read_edges(edges_path, node_ids)

    cm = coo_matrix((weights, (sources, targets)), shape=(len(node_ids), len(node_ids)))

    print("node ids", node_ids.shape)
    print("weights", cm.data.shape)
    print("rows", cm.row.shape)
    print("cols", cm.col.shape)

    G = cg.csrgraph(cm.tocsr(), node_ids, copy=False)

    embeddings = nodevectors.GGVec(
        n_components=dim,
        threads=n_threads,
        verbose=True,
    ).fit_transform(G)

    embedding_path = os.path.join(directory, f"high_embeddings-{dim}.npy")
    print("saving", embedding_path)
    np.save(embedding_path, embeddings)

if __name__ == "__main__":
    main()
