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

all: $(DATA)/directory.sqlite $(DATA)/atlas.sqlite
init: $(DATA)/nodes.arrow $(DATA)/edges.arrow
embeddings: $(DATA)/embeddings-$(DIM).npy
knn: $(DATA)/knn-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow
fss: $(DATA)/fss-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow
positions: $(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy

colors: $(DATA)/colors.npy
save: $(DATA)/atlas.sqlite

$(DATA)/graph.sqlite:
	exit 1

$(DATA)/directory.sqlite: $(DATA)/graph.sqlite
	sqlite3 $(DATA)/directory.sqlite 'CREATE TABLE users(id INTEGER PRIMARY KEY NOT NULL, did TEXT);'
	sqlite3 $(DATA)/directory.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO users(id, did) SELECT rowid, did FROM graph.nodes;'
	sqlite3 $(DATA)/directory.sqlite 'CREATE INDEX user_did ON users(did);'

$(DATA)/nodes.arrow $(DATA)/edges.arrow: $(DATA)/graph.sqlite
	python sqlite_to_arrow.py $(DATA)

$(DATA)/embeddings-$(DIM).npy: $(DATA)/nodes.arrow $(DATA)/edges.arrow
	python embedding.py $(DATA)

$(DATA)/knn-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow: $(DATA)/embeddings-$(DIM).npy
	python knn.py $(DATA)

$(DATA)/fss-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/knn-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow
	python fss.py $(DATA)

$(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/fss-$(DIM)-$(METRIC)-$(N_NEIGHBORS).arrow
	python positions.py $(DATA)

$(DATA)/cluster_labels-$(DIM)-$(N_CLUSTERS).npy $(DATA)/cluster_centers-$(DIM)-$(N_CLUSTERS).npy:  \
		$(DATA)/embeddings-$(DIM).npy
	python labels.py $(DATA)

$(DATA)/colors-$(DIM)-$(N_CLUSTERS).npy: \
		$(DATA)/incoming_degrees.npy \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/cluster_centers-$(DIM)-$(N_CLUSTERS).npy
	python colors.py $(DATA)

$(DATA)/atlas.sqlite: \
		$(DATA)/nodes.arrow \
		$(DATA)/colors.npy \
		$(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
	python save_graph.py $(DATA)

clean:
	rm -f $(DATA)/directory.sqlite
	rm -f $(DATA)/atlas.sqlite
	rm -f $(DATA)/*.npy
