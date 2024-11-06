import sys
import os

import pickle
import numpy as np
from scipy import sparse
from scipy import stats

import hdbscan

from dotenv import load_dotenv
load_dotenv()

from graph_utils import save_labels

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    label_path = os.path.join(directory, 'graph-label-{:d}-{:d}.pkl'.format(dim, n_neighbors))
    graph_database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))
    atlas_database_path = os.path.join(directory, 'atlas-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))

    print("label_path", label_path)
    print("graph_database_path", graph_database_path)
    print("atlas_database_path", atlas_database_path)

    with open(label_path, 'rb') as file:
        # (node_ids, labels, probabilities) = pickle.load(file)
        (node_ids, labels) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("labels:", type(labels), labels.shape)
    # print("probabilities:", type(probabilities), probabilities.shape)

    save_labels(graph_database_path, node_ids, labels)
    save_labels(atlas_database_path, node_ids, labels)

if __name__ == "__main__":
    main()
