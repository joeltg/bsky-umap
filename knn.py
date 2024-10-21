import pickle

import hnswlib
import numpy as np

EMBEDDINGS = 'graph-100000.emb.pkl'

dim = 32

def main():
    with open(EMBEDDINGS, 'rb') as file:
        (names, embeddings) = pickle.load(file)
    np.save("graph-100000.emb.npy", embeddings)

    # Declaring index
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

    # Initing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    p.init_index(max_elements=len(names), ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)

    # # Set number of threads used during batch search/construction
    # # By default using all available cores
    # p.set_num_threads(4)

    # print("Adding first batch of %d elements" % (len(embeddings)))
    p.add_items(embeddings)

    for id, emb in zip(names, embeddings):
        labels, distances = p.knn_query(emb, k=5)
        for label, dist in zip(labels[0], distances[0]):
            if dist == 0:
                continue
            print(id, label, 1/dist)

if __name__ == "__main__":
    main()
