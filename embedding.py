import sys
import os

import numpy as np

from ggvec import ggvec_main
from utils import NodeReader, EdgeReader

from dotenv import load_dotenv
load_dotenv()

def main():
    dim = int(os.environ['DIM'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    with NodeReader(nodes_path) as reader:
        (node_ids, incoming_degrees) = reader.get_nodes()

    edges_path = os.path.join(directory, "edges.arrow")
    with EdgeReader(edges_path) as reader:
        (weights, sources, targets) = reader.get_edges()

    print("node ids", node_ids.shape)
    print("weights", weights.shape)
    print("sources", sources.shape)
    print("targets", targets.shape)

    embeddings = ggvec_main(
        src = sources, dst = targets, data = weights,
        n_nodes = len(node_ids),
        n_components=dim,
        verbose=True,
    )

    embedding_path = os.path.join(directory, f"high_embeddings-{dim}.npy")
    print("saving", embedding_path)
    np.save(embedding_path, embeddings)

if __name__ == "__main__":
    main()
