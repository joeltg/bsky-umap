# import sys
# import os

# import sqlite3
# import pyarrow as pa

# from dotenv import load_dotenv
# load_dotenv()

# from graph_utils import coo_matrix_schema

# def main():
#     arguments = sys.argv[1:]
#     if len(arguments) == 0:
#         raise Exception("missing data directory")

#     directory = arguments[0]
#     database_path = os.path.join(directory, 'graph.sqlite')

#     (node_ids, data, rows, cols) = load_coo_matrix(database_path)
#     node_count = len(node_ids)
#     cm = coo_matrix((data, (rows, cols)), shape=(node_count, node_count))

#     # matrix_path = os.path.join(directory, "graph-coo-matrix.pkl")
#     # with open(matrix_path, 'wb') as file:
#     #     pickle.dump((cm, node_ids), file)

#     data = pa.array(cm.data, pa.float16())
#     row = pa.array(cm.row, pa.uint32())
#     col = pa.array(cm.col, pa.uint32())

#     print("data.type", data.type, pa.uint32())



#     coo_matrix_path = os.path.join(directory, "coo_matrix.arrow")

#     with pa.OSFile(coo_matrix_path, 'wb') as sink:
#         with pa.ipc.new_file(sink, schema=coo_matrix_schema) as writer:
#             batch = pa.record_batch([data, row, col], schema=coo_matrix_schema)
#             writer.write(batch)

#     id = pa.array(node_ids, pa.uint32())
#     node_ids_schema = pa.schema([
#         pa.field("id", id.type)
#     ])

#     node_ids_path = os.path.join(directory, "node_ids.arrow")
#     with pa.OSFile(node_ids_path, 'wb') as sink:
#         with pa.ipc.new_file(sink, schema=node_ids_schema) as writer:
#             batch = pa.record_batch([id], schema=node_ids_schema)
#             writer.write(batch)

# if __name__ == "__main__":
#     main()
