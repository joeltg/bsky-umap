import sys
import os

import pickle
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy import stats

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
    database_path = os.path.join(directory, 'graph-umap-{:d}-{:d}.sqlite'.format(dim, n_neighbors))

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("embeddings:", type(embeddings), embeddings.shape)

    print("Performing Truncated SVD (equivalent to PCA for this case)...")
    svd = TruncatedSVD(n_components=1, random_state=42)
    components = svd.fit_transform(embeddings)

    print("SVD completed.")
    print("Transformed data shape:", type(components), components.shape)

    hues = stats.norm.cdf(components, loc=np.mean(components), scale=np.std(components))  # -> uniform 0-1
    # hues = (hues * 255).astype(np.uint8)  # -> uniform 0-255

    # min = components.min();
    # max = components.max();
    # hues = np.clip(((components - min) / (max - min) * 255), 0, 255).astype(np.uint8)

    print("got hues:", type(hues), hues.shape)

    plot_distribution(hues)
    # save_colors(database_path, node_ids, hues)

if __name__ == "__main__":
    main()
