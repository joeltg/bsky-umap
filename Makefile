#!make
include .env

ifndef DATA
$(error DATA is not set)
endif

ifndef DIM
$(error DIM is not set)
endif

ifndef N_NEIGHBORS
$(error N_NEIGHBORS is not set)
endif

all: $(DATA)/directory.sqlite $(DATA)/atlas.sqlite
init: $(DATA)/nodes.arrow $(DATA)/edges.arrow
embeddings: $(DATA)/embeddings-$(DIM).npy
knn: $(DATA)/knn-$(DIM)-$(N_NEIGHBORS).arrow
fss: $(DATA)/fss-$(DIM)-$(N_NEIGHBORS).arrow
positions: $(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy

colors: $(DATA)/colors-$(DIM)-$(N_CLUSTERS).npy
save: $(DATA)/atlas.sqlite

$(DATA)/graph.sqlite:
	exit 1

$(DATA)/directory.sqlite: $(DATA)/graph.sqlite
	sqlite3 $(DATA)/directory.sqlite 'CREATE TABLE users(id INTEGER PRIMARY KEY NOT NULL, did TEXT);'
	sqlite3 $(DATA)/directory.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO users(id, did) SELECT rowid, did FROM graph.nodes;'
	sqlite3 $(DATA)/directory.sqlite 'CREATE INDEX user_did ON users(did);'

$(DATA)/ids.npy $(DATA)/sources.npy $(DATA)/targets.npy $(DATA)/incoming_degrees.npy $(DATA)/outgoing_degrees.npy: \
		$(DATA)/graph.sqlite
	python load_graph.py $(DATA)

$(DATA)/edges-csr-indices.npy $(DATA)/edges-csr-indptr.npy: \
		$(DATA)/ids.npy $(DATA)/sources.npy $(DATA)/targets.npy
	python save_csr.py $(DATA)

$(DATA)/edges-csc-indices.npy $(DATA)/edges-csc-indptr.npy: \
		$(DATA)/ids.npy $(DATA)/sources.npy $(DATA)/targets.npy
	python save_csc.py $(DATA)

$(DATA)/edges-csc-alias-probs.npy $(DATA)/edges-csc-alias-indices.npy: \
		$(DATA)/outgoing_degrees.npy \
		$(DATA)/edges-csc-indptr.npy \
		$(DATA)/edges-csc-indices.npy
	python save_csc_alias_tables.py $(DATA)

$(DATA)/edges-csr-alias-probs.npy $(DATA)/edges-csr-alias-indices.npy: \
		$(DATA)/incoming_degrees.npy \
		$(DATA)/edges-csr-indptr.npy \
		$(DATA)/edges-csr-indices.npy
	python save_csc_alias_tables.py $(DATA)

$(DATA)/mutual-edges-coo.npy $(DATA)/mutual-degrees.npy: \
		$(DATA)/edges-csc-indices.npy $(DATA)/edges-csc-indptr.npy \
		$(DATA)/edges-csr-indices.npy $(DATA)/edges-csr-indptr.npy
	python save_mutuals.py $(DATA)

$(DATA)/embeddings-$(DIM).npy: \
		$(DATA)/edges-csr-indices.npy $(DATA)/edges-csr-indptr.npy \
		$(DATA)/edges-csc-indices.npy $(DATA)/edges-csc-indptr.npy \
		$(DATA)/edges-csr-alias-probs.npy $(DATA)/edges-csr-alias-indices.npy \
		$(DATA)/edges-csc-alias-probs.npy $(DATA)/edges-csc-alias-indices.npy \
		$(DATA)/mutual-edges-coo.npy $(DATA)/mutual-degrees.npy
	python nnvec.py $(DATA)

$(DATA)/knn-$(DIM)-$(N_NEIGHBORS).arrow: $(DATA)/embeddings-$(DIM).npy
	python knn.py $(DATA)

$(DATA)/fss-$(DIM)-$(N_NEIGHBORS).arrow: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/knn-$(DIM)-$(N_NEIGHBORS).arrow
	python fss.py $(DATA)

$(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/fss-$(DIM)-$(N_NEIGHBORS).arrow
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
		$(DATA)/positions-$(DIM)-$(N_NEIGHBORS).npy
	python save_graph.py $(DATA)

clean:
	rm -f $(DATA)/directory.sqlite
	rm -f $(DATA)/atlas.sqlite
	rm -f $(DATA)/*.npy
	rm -f $(DATA)/*.arrow
