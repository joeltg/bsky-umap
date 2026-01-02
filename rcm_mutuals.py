import sys
from typing import cast

from scipy.sparse import csr_array

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    data = load(directory, "csr-data.npy")
    indices = load(directory, "csr-indices.npy")
    indptr = load(directory, "csr-indptr.npy")

    G = csr_array((data, indices, indptr))

    M = cast(csr_array, G.minimum(G.T).tocsr())
    M.eliminate_zeros()

    save(directory, "mutuals-weights.npy", M.data)
    save(directory, "mutuals-indices.npy", M.indices)
    save(directory, "mutuals-indptr.npy", M.indptr)
