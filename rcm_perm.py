import sys

from scipy.sparse import csr_array
from scipy.sparse.csgraph import reverse_cuthill_mckee

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    data = load(directory, "mutuals-weights.npy")
    indices = load(directory, "mutuals-indices.npy")
    indptr = load(directory, "mutuals-indptr.npy")

    M = csr_array((data, indices, indptr))

    perm = reverse_cuthill_mckee(M, symmetric_mode=True)
    save(directory, "perm.npy", perm)

    # save(directory, "ids.npy", ids[perm])
    # save(directory, "sources.npy", sources[perm])
    # save(directory, "targets.npy", targets[perm])
