import sys

import polars as pl

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    weights = load(directory, "weights.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    # Sort edges by source (primary) then target (secondary) for CSR representation
    print("Sorting edges...")
    df = pl.DataFrame(
        {
            "sources": sources,
            "targets": targets,
            "weights": weights,
        }
    )
    df = df.sort(["sources", "targets"])

    print("Edges sorted!")

    save(directory, "weights.npy", df["weights"].to_numpy())
    save(directory, "sources.npy", df["sources"].to_numpy())
    save(directory, "targets.npy", df["targets"].to_numpy())
