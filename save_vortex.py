import os
import sys

import pyarrow as pa
import vortex as vx


from utils import load, write_edges, write_nodes

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    ids = load(directory, "ids.npy")
    incoming_degrees = load(directory, "incoming_degrees.npy")
    outgoing_degrees = load(directory, "outgoing_degrees.npy")

    nodes_path = os.path.join(directory, "nodes.arrow")
    write_nodes(nodes_path, ids, incoming_degrees, outgoing_degrees)

    weights = load(directory, "weights.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    edges_path = os.path.join(directory, "edges.arrow")
    write_edges(edges_path, (weights, rows, cols))
