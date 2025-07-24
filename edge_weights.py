import sys

import numpy as np
import tqdm
from dotenv import load_dotenv

from utils import load, save

load_dotenv()


def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    rows = load(directory, "sources.npy")
    cols = load(directory, "targets.npy")
    incoming_degrees = load(directory, "incoming_degrees.npy")
    outgoing_degrees = load(directory, "outgoing_degrees.npy")

    assert len(rows) == len(cols)
    assert len(incoming_degrees) == len(outgoing_degrees)
    edge_count = len(rows)

    # normalize edge weights
    # w(u,v) = 64 * (
    #   min(ln(d_out(u)+1), ln(d_in(v)+1)) / max(ln(d_out(u)+1), ln(d_in(v)+1))
    # ) / ln((d_out(u)+1) * (d_in(v)+1))

    # Pre-compute log transforms once
    w_outgoing = np.log1p(outgoing_degrees + 1, dtype=np.float32)
    w_incoming = np.log1p(incoming_degrees + 1, dtype=np.float32)

    # Allocate final weights array
    weights = np.zeros(edge_count, dtype=np.float32)

    # Process in chunks to avoid memory issues
    chunk_size = 1_000_000
    for start_idx in tqdm.trange(0, edge_count, chunk_size, desc="normalizing weights"):
        end_idx = min(start_idx + chunk_size, edge_count)

        # Get chunk of row/col indices
        chunk_rows = rows[start_idx:end_idx]
        chunk_cols = cols[start_idx:end_idx]

        # Compute weights for this chunk directly into the weights array
        w_src = w_outgoing[chunk_rows]
        w_dst = w_incoming[chunk_cols]
        w_min = np.minimum(w_src, w_dst)
        w_max = np.maximum(w_src, w_dst)

        weights[start_idx:end_idx] = 64.0 * (w_min / w_max) / (w_src + w_dst)

    print("done!")
    save(directory, "weights.npy", weights)


if __name__ == "__main__":
    main()
