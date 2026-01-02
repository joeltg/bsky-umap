import sys

import numpy as np
from scipy.sparse import coo_array

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
    ).tocsr()

    save(directory, "csr-data.npy", G.data)
    save(directory, "csr-indices.npy", G.indices)
    save(directory, "csr-indptr.npy", G.indptr)
