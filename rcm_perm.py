import sys

from rcm import reverse_cuthill_mckee
from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    indices = load(directory, "edges-csr-indices.npy")
    indptr = load(directory, "edges-csr-indptr.npy")

    perm = reverse_cuthill_mckee(indices, indptr)

    save(directory, "perm.npy", perm)

    # save(directory, "ids.npy", ids[perm])
    # save(directory, "sources.npy", sources[perm])
    # save(directory, "targets.npy", targets[perm])
