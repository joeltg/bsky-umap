import sys
import os

import pickle
import numpy as np
from scipy import sparse
from scipy import stats

import hdbscan

from dotenv import load_dotenv
load_dotenv()

from plot_distribution import plot_distribution

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    # label_path = os.path.join(directory, 'graph-label-{:d}.pkl'.format(dim))
    label_path = os.path.join(directory, 'graph-label-{:d}-{:d}.pkl'.format(dim, n_neighbors))
    database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))

    print("label_path", label_path)
    print("database_path", database_path)

    with open(label_path, 'rb') as file:
        (node_ids, labels, probabilities) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("labels:", type(labels), labels.shape)
    print("probabilities:", type(probabilities), probabilities.shape)

    plot_distribution(labels[labels != -1])

if __name__ == "__main__":
    main()
