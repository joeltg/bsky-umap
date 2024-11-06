import sys
import os

import pickle
import numpy as np
from scipy import sparse
from scipy import stats
from sklearn.cluster import KMeans

import hdbscan

from dotenv import load_dotenv
load_dotenv()

from plot_distribution import plot_distribution
from graph_utils import save_colors

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    # label_path = os.path.join(directory, 'graph-label-{:d}.pkl'.format(dim))
    label_path = os.path.join(directory, 'graph-label-{:d}-{:d}.pkl'.format(dim, n_neighbors))
    # database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))
    # embedding_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.pkl'.format(dim, n_neighbors))

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("embeddings:", type(embeddings), embeddings.shape)

    print("Performing k-means clustering")
    clusterer = KMeans(
        n_clusters=1000,
        verbose=1
    ).fit(embeddings)
    print("k-means clustering completed.")

    labels = clusterer.labels_
    print("labels", type(labels), labels.shape)
    # probabilities = clusterer.probabilities_
    # print("probabilities", type(probabilities), probabilities.shape)

    with open(label_path, 'wb') as file:
        pickle.dump((node_ids, labels), file)

    # plot_distribution(hues)
    # save_colors(database_path, node_ids, hues)

if __name__ == "__main__":
    main()
