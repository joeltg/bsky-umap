import sys
import os

import pickle
from sklearn.cluster import KMeans

from dotenv import load_dotenv
load_dotenv()

def main():
    dim = int(os.environ['DIM'])
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    n_clusters = int(os.environ['N_CLUSTERS'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]
    embedding_path = os.path.join(directory, 'graph-emb-{:d}.pkl'.format(dim))
    label_path = os.path.join(directory, 'graph-label-{:d}-{:d}.pkl'.format(dim, n_neighbors))

    with open(embedding_path, 'rb') as file:
        (node_ids, embeddings) = pickle.load(file)

    print("node_ids:", type(node_ids), node_ids.shape)
    print("embeddings:", type(embeddings), embeddings.shape)

    print("Performing k-means clustering")
    clusterer = KMeans(
        n_clusters=n_clusters,
        verbose=1,
    ).fit(embeddings)
    print("k-means clustering completed.")

    labels = clusterer.labels_
    print("labels", type(labels), labels.shape)

    cluster_centers = clusterer.cluster_centers_
    print("cluster_centers", type(cluster_centers), cluster_centers.shape)

    with open(label_path, 'wb') as file:
        pickle.dump((node_ids, labels, cluster_centers), file)

if __name__ == "__main__":
    main()
