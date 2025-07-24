import os
import sys

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

from ggvec import ggvec_main
from utils import load, save

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

    n_threads: None | int = None
    if "N_THREADS" in os.environ:
        n_threads = int(os.environ["N_THREADS"])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    ids: NDArray[np.uint32] = load(directory, "ids.npy")
    sources: NDArray[np.uint32] = load(directory, "sources.npy")
    targets: NDArray[np.uint32] = load(directory, "targets.npy")
    weights: NDArray[np.float32] = load(directory, "weights.npy")

    embeddings = ggvec_main(
        src=sources,
        dst=targets,
        data=weights,
        n_nodes=len(ids),
        n_components=dim,
        n_threads=n_threads,
        **ggvec_kwargs,
    )

    save(directory, f"embeddings-{dim}.npy", embeddings)


if __name__ == "__main__":
    main()
