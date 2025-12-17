import os
import sys

import pyarrow as pa
import vortex as vx

# import matplotlib.pyplot as plt
from utils import load

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]
    # input_path = arguments[0]
    # output_path = arguments[1]

    ids = load(directory, "ids.npy")
    incoming_degrees = load(directory, "incoming_degrees.npy")
    outgoing_degrees = load(directory, "outgoing_degrees.npy")

    weights = load(directory, "weights.npy")
    sources = load(directory, "sources.npy")
    targets = load(directory, "targets.npy")

    # Sort edges by source (primary) then target (secondary) for CSR representation
    print("Sorting edges...")
    edge_table = pa.table({"sources": sources, "targets": sources, "weights": weights})

    sorted_table = edge_table.sort_by(
        [("sources", "ascending"), ("targets", "ascending")]
    )

    sources = sorted_table["sources"].to_numpy()
    targets = sorted_table["targets"].to_numpy()
    weights = sorted_table["weights"].to_numpy()
    print("Edges sorted!")

    node_data = vx.Array.from_arrow(
        pa.StructArray.from_arrays(
            arrays=[ids, incoming_degrees, outgoing_degrees],
            fields=[
                pa.field("ids", pa.uint32(), nullable=False),
                pa.field("incoming_degrees", pa.uint32(), nullable=False),
                pa.field("outgoing_degrees", pa.uint32(), nullable=False),
            ],
        ),
    )

    assert node_data.dtype == vx.struct(
        {
            "ids": vx.uint(32),
            "incoming_degrees": vx.uint(32),
            "outgoing_degrees": vx.uint(32),
        }
    )

    node_output_path = os.path.join(directory, "nodes.vortex")
    vx.io.write(node_data, node_output_path)

    edge_data = vx.Array.from_arrow(
        pa.StructArray.from_arrays(
            arrays=[weights, sources, targets],
            fields=[
                pa.field("weights", pa.float32(), nullable=False),
                pa.field("sources", pa.int32(), nullable=False),
                pa.field("targets", pa.int32(), nullable=False),
            ],
        ),
    )

    assert edge_data.dtype == vx.struct(
        {
            "weights": vx.float_(32),
            "sources": vx.int_(32),
            "targets": vx.int_(32),
        }
    )

    edge_output_path = os.path.join(directory, "edges.vortex")
    vx.io.write(edge_data, edge_output_path)
