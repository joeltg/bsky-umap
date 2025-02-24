import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import sqlite3

edge_schema = pa.schema([
    pa.field('weight', pa.float16()),
    pa.field('source', pa.uint32()),
    pa.field('target', pa.uint32()),
])

def write_edges(path: str, edges: tuple[NDArray[np.float16], NDArray[np.uint32], NDArray[np.uint32]]):
    (weights, sources, targets) = edges

    data = [
        pa.array(weights, type=pa.float16()),
        pa.array(sources, type=pa.uint32()),
        pa.array(targets, type=pa.uint32()),
    ]

    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema=edge_schema) as writer:
            batch = pa.record_batch(data, schema=edge_schema)
            writer.write(batch)

def read_edges(path: str, node_ids: NDArray[np.uint32]) -> tuple[NDArray[np.float16], NDArray[np.uint32], NDArray[np.uint32]]:
    with pa.memory_map(path, 'r') as source:
        data = pa.ipc.open_file(source).read_all()

    weights: NDArray[np.float16] = data['weight'].to_numpy()
    sources: NDArray[np.uint32] = data['source'].to_numpy()
    targets: NDArray[np.uint32] = data['target'].to_numpy()

    assert len(weights) == len(sources)
    assert len(weights) == len(targets)

    return (weights, sources, targets)

node_schema = pa.schema([
    pa.field('id', pa.uint32()),
    pa.field('incoming_degree', pa.uint32()),
])

def write_nodes(path: str, ids: NDArray[np.uint32], incoming_degrees: NDArray[np.uint32]):
    data = [
        pa.array(ids, type=pa.uint32()),
        pa.array(incoming_degrees, type=pa.uint32()),
    ]

    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema=node_schema) as writer:
            batch = pa.record_batch(data, schema=node_schema)
            writer.write(batch)

def read_nodes(path: str) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    with pa.memory_map(path, 'r') as source:
        data = pa.ipc.open_file(source).read_all()

    ids: NDArray[np.uint32] = data['id'].to_numpy()
    incoming_degrees: NDArray[np.uint32] = data['incoming_degree'].to_numpy()
    return (ids, incoming_degrees)

def save_csr_graph(database_path, node_ids, csrgraph):
    coograph = csrgraph.tocoo()
    coograph.sum_duplicates()
    coograph.eliminate_zeros()

    if coograph.shape[0] != len(node_ids) or coograph.shape[1] != len(node_ids):
        raise Exception("unexpected matrix dimensions")

    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        id INTEGER PRIMARY KEY,
        x FLOAT NOT NULL DEFAULT 0,
        y FLOAT NOT NULL DEFAULT 0,
        mass INTEGER NOT NULL DEFAULT 0,
        color INTEGER NOT NULL DEFAULT 0
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS edges (
        source INTEGER NOT NULL,
        target INTEGER NOT NULL
    );
    ''')

    try:
        conn.execute("BEGIN")
        cursor.execute("DELETE FROM nodes")
        cursor.execute("DELETE FROM edges")

        data = [(id) for id in zip(node_ids)]
        cursor.executemany("INSERT INTO nodes(rowid) VALUES (?)", data)

        data = [(int(node_ids[s]), int(node_ids[t]), float(w)) for s, t, w in zip(coograph.row, coograph.col, coograph.data)]
        cursor.executemany("INSERT INTO edges(source, target, weight) VALUES (?, ?, ?)", data)

        cursor.execute("CREATE INDEX IF NOT EXISTS edge_source ON edges(source)")
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"SQLite error: {e}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def save_colors(database_path, node_ids, colors):
    print("opening", database_path)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")
        data = [(int(color), id) for id, color in zip(node_ids, colors)]
        cursor.executemany("UPDATE nodes SET color = ? WHERE id = ?", data)
        conn.commit()

    except sqlite3.Error as e:
        conn.rollback()
        raise sqlite3.Error(f"SQLite error: {e}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
