import os
import sys

import numba
import scipy
from dotenv import load_dotenv

from ggvec.ggvec import ggvec_main
from utils import EdgeReader, NodeReader, save

load_dotenv()


def main():
    dim = int(os.environ["DIM"])

    # Build kwargs for ggvec parameters from environment variables
    ggvec_kwargs = {}
    if "LEARNING_RATE" in os.environ:
        ggvec_kwargs["learning_rate"] = float(os.environ["LEARNING_RATE"])
    if "NEGATIVE_RATIO" in os.environ:
        ggvec_kwargs["negative_ratio"] = float(os.environ["NEGATIVE_RATIO"])
    if "NEGATIVE_DECAY" in os.environ:
        ggvec_kwargs["negative_decay"] = float(os.environ["NEGATIVE_DECAY"])
    if "MAX_LOSS" in os.environ:
        ggvec_kwargs["max_loss"] = float(os.environ["MAX_LOSS"])
    if "MAX_EPOCH" in os.environ:
        ggvec_kwargs["max_epoch"] = int(os.environ["MAX_EPOCH"])
    if "TOL" in os.environ:
        ggvec_kwargs["tol"] = float(os.environ["TOL"])
    if "TOL_SAMPLES" in os.environ:
        ggvec_kwargs["tol_samples"] = int(os.environ["TOL_SAMPLES"])

    if "N_THREADS" in os.environ:
        numba.set_num_threads(int(os.environ["N_THREADS"]))

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    with NodeReader(nodes_path) as reader:
        (ids, incoming_degrees, outgoing_degrees) = reader.get_nodes()

    edges_path = os.path.join(directory, "edges.arrow")
    with EdgeReader(edges_path) as reader:
        (weights, sources, targets) = reader.get_edges()

    G = scipy.sparse.coo_array(
        (weights, (sources, targets)), shape=(len(ids), len(ids))
    )

    embeddings = ggvec_main(G, n_components=dim, **ggvec_kwargs)

    save(directory, f"embeddings-{dim}.npy", embeddings)


if __name__ == "__main__":
    main()
