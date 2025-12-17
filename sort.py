import sys

import polars as pl

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    # Sort edges by source (primary) then target (secondary) for CSR representation
    print("Sorting edges...")
    df = pl.DataFrame(
        {
            "sources": sources,
            "targets": targets,
        }
    )
    df = df.sort(["sources", "targets"])

    print("Edges sorted!")

    save(directory, "sources.npy", df["sources"].to_numpy())
    save(directory, "targets.npy", df["targets"].to_numpy())
