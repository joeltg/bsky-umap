import sys
import os

import sqlite3
import numpy as np
from numpy.typing import NDArray

from quadtree.quadtree import QuadTree
from utils import read_nodes
from dotenv import load_dotenv

load_dotenv()

def get_side_len(positions: NDArray[np.float32]) -> float:
    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])
    print(f"min_x: {min_x}, max_x: {max_x}")

    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])
    print(f"min_y: {min_y}, max_y: {max_y}")

    s = 2 * max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
    return s.astype(float)

class Engine:
    mass: NDArray[np.float32]
    positions: NDArray[np.float32]
    forces: NDArray[np.float32]
    trees: list[QuadTree]

    def __init__(self, positions: NDArray[np.float32], mass: NDArray[np.float32]):
        self.mass = mass
        self.positions = positions
        self.forces = np.zeros((len(positions), 2), dtype=np.float32)
        self.trees = [QuadTree(0, (0.0, 0.0))]

    def build(self):
        side_len = get_side_len(self.positions)
        self.trees[0].reset(side_len, (0.0, 0.0))

        for i in range(len(self.positions)):
            x = self.positions[i][0].astype(float)
            y = self.positions[i][1].astype(float)
            self.trees[0].insert((x, y))

    def tick(self):
        self.build()
        self.update_nodes(0, len(self.positions))
        print("avg force", np.average(np.linalg.norm(self.forces)))

    def update_nodes(self, start: int, end: int, temperature: float = 1.0):
        for i in range(start, end):
            if i >= len(self.positions):
                break

            x = self.positions[i][0].astype(float)
            y = self.positions[i][1].astype(float)
            mass = self.mass[i].astype(float)

            (dx, dy) = self.trees[0].get_force((x, y), mass);
            self.forces[i][0] = dx
            self.forces[i][1] = dy
            self.positions[i][0] += dx * temperature;
            self.positions[i][1] += dy * temperature;


def main():
    n_neighbors = int(os.environ['N_NEIGHBORS'])
    n_threads = int(os.environ['N_THREADS'])
    dim = int(os.environ['DIM'])

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        raise Exception("missing data directory")

    directory = arguments[0]

    nodes_path = os.path.join(directory, "nodes.arrow")
    (ids, incoming_degrees) = read_nodes(nodes_path)
    # incoming_degrees_norm: NDArray[np.float32] = incoming_degrees.astype(np.float32) / np.max(incoming_degrees).astype(np.float32)
    # node_mass: NDArray[np.float32] = 100 * np.sqrt(incoming_degrees_norm).astype(np.float32)

    # low_embeddings_path = os.path.join(directory, f"low_embeddings-{dim}-{n_neighbors}.npy")
    # low_embeddings: NDArray[np.float32] = np.load(low_embeddings_path)
    # print("loaded low_embeddings", low_embeddings_path, low_embeddings.shape)

    positions = np.zeros((len(ids), 2), dtype=np.float32)
    # positions_x = np.zeros(len(ids), dtype=np.float32)
    # positions_y = np.zeros(len(ids), dtype=np.float32)

    # engine = Engine(low_embeddings, node_mass)
    # print(engine)
    # engine.tick()
    # engine.tick()
    # engine.tick()
    # engine.tick()
    # engine.tick()
    # engine.tick()
    # engine.tick()

    database_path = os.path.join(directory, 'positions.sqlite')
    conn = sqlite3.connect(database_path)

    try:
        # Fill indices array
        cursor = conn.cursor()
        cursor.execute("SELECT id, x, y FROM nodes ORDER BY id ASC")
        for i, (id, x, y) in enumerate(cursor):
            positions[i][0] = x
            positions[i][1] = y
            # positions_x[i] = x
            # positions_y[i] = y
    finally:
        conn.close()

    path = os.path.join(directory, "positions.buffer")
    positions.tofile(path)
    # x_path = os.path.join(directory, "positions_y.buffer")
    # y_path = os.path.join(directory, "positions_x.buffer")
    # positions_x.tofile(x_path)
    # positions_y.tofile(y_path)

if __name__ == "__main__":
    main()
