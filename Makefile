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
init: $(DATA)/ids.npy $(DATA)/incoming_degrees.npy $(DATA)/outgoing_degrees.npy \
		$(DATA)/sources.npy $(DATA)/targets.npy $(DATA)/weights.npy
embeddings: $(DATA)/embeddings-$(DIM).npy
knn: $(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy \
		$(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
fss: $(DATA)/fss_rows-$(DIM)-$(METRIC)-$(n_neighbors).npy \
		$(DATA)/fss_cols-$(DIM)-$(METRIC)-$(n_neighbors).npy \
		$(DATA)/fss_vals-$(DIM)-$(METRIC)-$(n_neighbors).npy
colors: $(DATA)/colors.npy
umap: $(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
save: $(DATA)/atlas.sqlite

$(DATA)/graph.sqlite:
	exit 1

$(DATA)/directory.sqlite: $(DATA)/graph.sqlite
	sqlite3 $(DATA)/directory.sqlite 'CREATE TABLE users(id INTEGER PRIMARY KEY NOT NULL, did TEXT);'
	sqlite3 $(DATA)/directory.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO users(id, did) SELECT rowid, did FROM graph.nodes;'
	sqlite3 $(DATA)/directory.sqlite 'CREATE INDEX user_did ON users(did);'

$(DATA)/ids.npy $(DATA)/incoming_degrees.npy $(DATA)/outgoing_degrees.npy $(DATA)/sources.npy $(DATA)/targets.npy: $(DATA)/graph.sqlite
	python load_graph.py $(DATA)

$(DATA)/weights.npy : $(DATA)/incoming_degrees.npy $(DATA)/outgoing_degrees.npy $(DATA)/sources.npy $(DATA)/targets.npy
	python edge_weights.py $(DATA)

$(DATA)/embeddings-$(DIM).npy: $(DATA)/ids.npy $(DATA)/sources.npy $(DATA)/targets.npy $(DATA)/weights.npy
	python embedding.py $(DATA)

$(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy $(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: \
		$(DATA)/embeddings-$(DIM).npy
	python knn.py $(DATA)

$(DATA)/fss_rows-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy $(DATA)/fss_cols-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy $(DATA)/fss_vals-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy  \
		$(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
	python fuzzy_simplicial_set.py $(DATA)

$(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/fss_rows-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy \
		$(DATA)/fss_cols-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy \
		$(DATA)/fss_vals-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
	PYTHONPATH=umap-learn python project.py $(DATA)

# $(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy: \
# 		$(DATA)/embeddings-$(DIM).npy \
# 		$(DATA)/knn_indices-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy  \
# 		$(DATA)/knn_dists-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
# 	PYTHONPATH=umap-learn python project.py $(DATA)

$(DATA)/cluster_labels-$(DIM)-$(N_CLUSTERS).npy $(DATA)/cluster_centers-$(DIM)-$(N_CLUSTERS).npy:  \
		$(DATA)/embeddings-$(DIM).npy
	python labels.py $(DATA)

$(DATA)/colors.npy: \
		$(DATA)/incoming_degrees.npy \
		$(DATA)/embeddings-$(DIM).npy \
		$(DATA)/cluster_centers-$(DIM)-$(N_CLUSTERS).npy
	python colors.py $(DATA)

$(DATA)/atlas.sqlite: $(DATA)/ids.npy $(DATA)/colors.npy $(DATA)/positions-$(DIM)-$(METRIC)-$(N_NEIGHBORS).npy
	python save_graph.py $(DATA)

clean:
	rm -f $(DATA)/directory.sqlite
	rm -f $(DATA)/atlas.sqlite
	rm -f $(DATA)/*.npy
