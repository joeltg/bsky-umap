import sys

import pyarrow as pa
import vortex as vx

# import matplotlib.pyplot as plt
from utils import EdgeReader

if __name__ == "__main__":
    arguments = sys.argv[1:]
    input_path = arguments[0]
    output_path = arguments[0]

    with EdgeReader(input_path) as reader:
        (weights, sources, targets) = reader.get_edges()

    print("weights", weights)
    print("sources", sources)
    print("targets", targets)

    # plt.hist(weights, bins=500)
    # plt.show()

    dtype = vx.struct(
        {
            "weights": vx.float_(32),
            "sources": vx.int_(32),
            "targets": vx.int_(32),
        }
    )

    data = vx.Array.from_arrow(
        pa.StructArray.from_arrays(
            arrays=[weights, sources, targets],
            fields=[
                pa.field("weights", pa.float32(), nullable=False),
                pa.field("sources", pa.int32(), nullable=False),
                pa.field("targets", pa.int32(), nullable=False),
            ],
        ),
    )

    print("data", data, dtype, data.dtype == dtype)

    vx.io.write(data, output_path)
