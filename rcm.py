import sys

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import reverse_cuthill_mckee

# import matplotlib.pyplot as plt
from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    ids = load(directory, "ids.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    G = coo_array(
        (np.ones(len(sources), dtype=np.float32), (sources, targets)),
        shape=(len(ids), len(ids)),
    )

    perm = reverse_cuthill_mckee(G.multiply(G.T).tocsr(), symmetric_mode=True)
    save(directory, "perm.npy", perm)

    # save(directory, "ids.npy", ids[perm])
    # save(directory, "sources.npy", sources[perm])
    # save(directory, "targets.npy", targets[perm])
