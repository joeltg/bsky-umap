import sys

import numpy as np
from numpy.typing import NDArray

from alias_table import build_all_alias_tables
from utils import load, save

if __name__ == "__main__":
    arguments = sys.argv[1:]
    directory = arguments[0]

    incoming_degrees: NDArray[np.uint32] = load(directory, "incoming_degrees.npy")
    # csr_indptr: NDArray[np.int64] = load_array(directory, "edges-csr-indptr.vortex")
    # csr_indices: NDArray[np.int32] = load_array(directory, "edges-csr-indices.vortex")
    csr_indptr: NDArray[np.int64] = load(directory, "edges-csr-indptr.npy")
    csr_indices: NDArray[np.int32] = load(directory, "edges-csr-indices.npy")

    print("computing csr alias table")
    csr_alias_probs, csr_alias_indices = build_all_alias_tables(
        csr_indptr, csr_indices, incoming_degrees
    )

    csr_alias_probs = (csr_alias_probs * 0x10000).astype(np.uint16)
    # save_array(directory, "edges-csr-alias-probs.vortex", csr_alias_probs)
    # save_array(directory, "edges-csr-alias-indices.vortex", csr_alias_indices)
    save(directory, "edges-csr-alias-probs.npy", csr_alias_probs)
    save(directory, "edges-csr-alias-indices.npy", csr_alias_indices)
