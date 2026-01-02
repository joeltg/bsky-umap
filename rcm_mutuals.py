import sys
from typing import cast

import numpy as np
from scipy.sparse import coo_array, csr_array

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

    M = cast(csr_array, G.multiply(G.T).tocsr())

    save(directory, "mutuals-weights.npy", M.data)
    save(directory, "mutuals-indices.npy", M.indices)
    save(directory, "mutuals-indptr.npy", M.indptr)
