import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

edge_schema = pa.schema([
    pa.field('weight', pa.float32()),
    pa.field('source', pa.uint32()),
    pa.field('target', pa.uint32()),
])

def write_edges(path: str, edges: tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]):
    (weights, sources, targets) = edges

    data = [
        pa.array(weights, type=pa.float32()),
        pa.array(sources, type=pa.uint32()),
        pa.array(targets, type=pa.uint32()),
    ]

    with pa.OSFile(path, 'wb') as sink:
        with pa.ipc.new_file(sink, schema=edge_schema) as writer:
            batch = pa.record_batch(data, schema=edge_schema)
            writer.write(batch)

def read_edges(path: str, node_ids: NDArray[np.uint32]) -> tuple[NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]:
    with pa.memory_map(path, 'r') as source:
        data = pa.ipc.open_file(source).read_all()

    weights: NDArray[np.float32] = data['weight'].to_numpy()
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
