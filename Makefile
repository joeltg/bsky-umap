#!make
include .env

ifndef DATA
$(error DATA is not set)
endif

ifndef DIM
$(error DIM is not set)
endif

ifndef METRIC
$(error METRIC is not set)
endif

ifndef N_NEIGHBORS
$(error N_NEIGHBORS is not set)
endif

all: colors umap
init: $(DATA)/directory.sqlite $(DATA)/edges.arrow $(DATA)/nodes.arrow $(DATA)/ids.buffer
embeddings: $(DATA)/embeddings-$(DIM).npy
colors: $(DATA)/colors.buffer
umap: $(DATA)/positions.sqlite
save: $(DATA)/positions.buffer $(DATA)/atlas.sqlite

$(DATA)/graph.sqlite:
	exit 1

$(DATA)/directory.sqlite: $(DATA)/graph.sqlite
	sqlite3 $(DATA)/directory.sqlite 'CREATE TABLE users(id INTEGER PRIMARY KEY NOT NULL, did TEXT);'
	sqlite3 $(DATA)/directory.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO users(id, did) SELECT rowid, did FROM graph.nodes;'
	sqlite3 $(DATA)/directory.sqlite 'CREATE INDEX user_did ON users(did);'

# $(DATA)/ids.buffer: $(DATA)/graph.sqlite
# 	python sqlite_to_ids.py $(DATA)

$(DATA)/edges.arrow $(DATA)/nodes.arrow $(DATA)/ids.buffer: $(DATA)/graph.sqlite
	python sqlite_to_arrow.py $(DATA)

$(DATA)/embeddings-$(DIM).npy: $(DATA)/nodes.arrow $(DATA)/edges.arrow
	python embedding.py $(DATA)

$(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy $(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: $(DATA)/embeddings-$(DIM).npy
	python knn.py $(DATA)

$(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy: $(DATA)/embeddings-$(DIM).npy $(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy $(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
	PYTHONPATH=umap-learn python project.py $(DATA)

$(DATA)/positions.buffer: $(DATA)/positions.sqlite $(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy
	python anneal.py $(DATA)

$(DATA)/positions.sqlite: $(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy
	python save_graph.py $(DATA)

$(DATA)/cluster_labels-$(DIM)-$(N_NEIGHBORS)-$(N_CLUSTERS).npy $(DATA)/cluster_centers-$(DIM)-$(N_NEIGHBORS)-$(N_CLUSTERS).npy: $(DATA)/embeddings-$(DIM).npy
	python labels.py $(DATA)

$(DATA)/colors.buffer: $(DATA)/nodes.arrow $(DATA)/embeddings-$(DIM).npy $(DATA)/cluster_labels-$(DIM)-$(N_NEIGHBORS)-$(N_CLUSTERS).npy $(DATA)/cluster_centers-$(DIM)-$(N_NEIGHBORS)-$(N_CLUSTERS).npy
	python colors.py $(DATA)

$(DATA)/atlas.sqlite: $(DATA)/positions.sqlite
	sqlite3 $(DATA)/atlas.sqlite 'CREATE VIRTUAL TABLE nodes USING rtree(id INTEGER PRIMARY KEY, minX FLOAT NOT NULL, maxX FLOAT NOT NULL, minY FLOAT NOT NULL, maxY FLOAT NOT NULL);'
	sqlite3 $(DATA)/atlas.sqlite 'ATTACH DATABASE "$(DATA)/positions.sqlite" AS graph; INSERT INTO nodes(id, minX, maxX, minY, maxY) SELECT rowid, x, x, y, y FROM graph.nodes;'

clean:
	rm -f $(DATA)/directory.sqlite
	rm -f $(DATA)/positions.sqlite
	rm -f $(DATA)/atlas.sqlite
	rm -f $(DATA)/*.arrow
	rm -f $(DATA)/*.npy
	rm -f $(DATA)/*.buffer
