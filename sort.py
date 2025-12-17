import sys

import pyarrow as pa

from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    weights = load(directory, "weights.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    # Sort edges by source (primary) then target (secondary) for CSR representation
    print("Sorting edges...")
    edge_table = pa.table({"sources": sources, "targets": sources, "weights": weights})

    sorted_table = edge_table.sort_by(
        [("sources", "ascending"), ("targets", "ascending")]
    )

    print("Edges sorted!")

    save(directory, "weights.npy", sorted_table["weights"].to_numpy())
    save(directory, "sources.npy", sorted_table["sources"].to_numpy())
    save(directory, "targets.npy", sorted_table["targets"].to_numpy())
