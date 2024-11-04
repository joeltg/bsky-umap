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
from graph_utils import save_colors

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    label_path = os.path.join(directory, 'graph-label-{:d}.pkl'.format(dim))
    # database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("embeddings:", type(embeddings), embeddings.shape)\

    clusterer = hdbscan.HDBSCAN(
        # min_cluster_size=100,
        core_dist_n_jobs=12,
        min_samples=100,  # Higher value to reduce memory
        algorithm='best',  # More memory efficient
        # leaf_size=100,  # Larger leaves = less memory overhead
        cluster_selection_method='leaf',
        cluster_selection_epsilon=0.15,
        metric='euclidean'  # Efficient metric
    )

    print("Performing HDBSCAN")
    clusterer.fit(embeddings)

    print("HDBSCAN completed.")
    print("Transformed data shape:", type(clusterer.labels_), clusterer.labels_.shape)

    with open(label_path, 'wb') as file:
        pickle.dump((node_ids, clusterer.labels_), file)

    # hues = stats.norm.cdf(components, loc=np.mean(components), scale=np.std(components))  # -> uniform 0-1
    # hues = (hues * 255).astype(np.uint8)  # -> uniform 0-255

    # min = components.min();
    # max = components.max();
    # hues = np.clip(((components - min) / (max - min) * 255), 0, 255).astype(np.uint8)

    # print("got hues:", type(hues), hues.shape)

    # plot_distribution(hues)
    # save_colors(database_path, node_ids, hues)

if __name__ == "__main__":
    main()
