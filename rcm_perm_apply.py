import sys

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    ids = load(directory, "ids.npy", copy=True)
    sources = load(directory, "sources.npy", copy=True)
    targets = load(directory, "targets.npy", copy=True)

    perm = load(directory, "perm.npy")

    save(directory, "ids.npy", ids[perm])
    save(directory, "sources.npy", sources[perm])
    save(directory, "targets.npy", targets[perm])
