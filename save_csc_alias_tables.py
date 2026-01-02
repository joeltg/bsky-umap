import sys

import numpy as np
from numpy.typing import NDArray

from alias_table import build_all_alias_tables
from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    outgoing_degrees: NDArray[np.uint32] = load(directory, "outgoing_degrees.npy")
    # csc_indptr: NDArray[np.int64] = load_array(directory, "edges-csc-indptr.vortex")
    # csc_indices: NDArray[np.int32] = load_array(directory, "edges-csc-indices.vortex")
    csc_indptr: NDArray[np.int64] = load(directory, "edges-csc-indptr.npy")
    csc_indices: NDArray[np.int32] = load(directory, "edges-csc-indices.npy")

    print("computing csc alias table")
    csc_alias_probs, csc_alias_indices = build_all_alias_tables(
        csc_indptr, csc_indices, outgoing_degrees
    )

    csc_alias_probs = (csc_alias_probs * 65536.0).astype(np.uint16)
    # save_array(directory, "edges-csc-alias-probs.vortex", csc_alias_probs)
    # save_array(directory, "edges-csc-alias-indices.vortex", csc_alias_indices)
    save(directory, "edges-csc-alias-probs.npy", csc_alias_probs)
    save(directory, "edges-csc-alias-indices.npy", csc_alias_indices)
