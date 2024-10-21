import pickle
import sqlite3
import numpy as np

import matplotlib.pyplot as plt

import umap

# DATABASE = '/Users/joelgustafson/Downloads/graph-100000.sqlite'
DATABASE = 'graph-100000.sqlite'
EMBEDDINGS = 'graph-100000.emb.pkl'

# EMBEDDINGS = 'graph-1000.emb.pkl'

dim = 32

def main():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        x FLOAT NOT NULL DEFAULT 0,
        y FLOAT NOT NULL DEFAULT 0
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS edges (
        source INTEGER NOT NULL,
        target INTEGER NOT NULL
    );
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS edge_source ON edges(source);')
    # cursor.execute('DELETE FROM EDGES;')

    # Unpickle the tuple
    with open(EMBEDDINGS, 'rb') as file:
        (names, embeddings) = pickle.load(file)

    reducer = umap.UMAP(verbose=True)
    result = reducer.fit_transform(embeddings)

    # print(result.shape)
    # print(result)

    # Prepare the data for insertion
    scale = 1000
    data = [(int(id), float(p[0] * scale), float(p[1] * scale)) for id, p in zip(names, result)]

    # Insert the data into the table
    cursor.executemany('''
    INSERT INTO nodes (rowid, x, y) VALUES (?, ?, ?)
        ON CONFLICT(rowid) DO UPDATE SET x = excluded.x, y = excluded.y
    ''', data)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    # plt.scatter(result[:, 0], result[:, 1])
    # plt.gca().set_aspect('equal', 'datalim')
    # # plt.title('UMAP projection of the Penguin dataset', fontsize=24);
    # plt.show()

if __name__ == "__main__":
    main()
