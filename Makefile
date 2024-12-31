DATA        := ./data
DIM         := 32
N_NEIGHBORS := 25

all: $(DATA)/graph-umap-$(DIM)-$(N_NEIGHBORS).sqlite $(DATA)/colors.sqlite $(DATA)/atlas.sqlite

$(DATA)/graph-coo-matrix.pkl:
	python load_graph.py $(DATA)

$(DATA)/graph-emb-$(DIM).pkl: $(DATA)/graph-coo-matrix.pkl
	DIM=$(DIM) python embedding.py $(DATA)

$(DATA)/graph-knn-$(DIM)-$(N_NEIGHBORS).pkl: $(DATA)/graph-emb-$(DIM).pkl
	DIM=$(DIM) N_NEIGHBORS=$(N_NEIGHBORS) python knn.py $(DATA)

$(DATA)/graph-umap-$(DIM)-$(N_NEIGHBORS).pkl $(DATA)/graph-umap-$(DIM)-$(N_NEIGHBORS).sqlite: $(DATA)/graph-knn-$(DIM)-$(N_NEIGHBORS).pkl $(DATA)/graph-emb-$(DIM).pkl
	DIM=$(DIM) N_NEIGHBORS=$(N_NEIGHBORS) python project.py $(DATA)

$(DATA)/graph-label-$(DIM)-$(N_NEIGHBORS).pkl: $(DATA)/graph-emb-$(DIM).pkl
	DIM=$(DIM) N_NEIGHBORS=$(N_NEIGHBORS) N_CLUSTERS=25 python labels.py $(DATA)

$(DATA)/colors.sqlite: $(DATA)/graph-umap-$(DIM)-$(N_NEIGHBORS).pkl $(DATA)/graph-label-$(DIM)-$(N_NEIGHBORS).pkl
	sqlite3 $(DATA)/colors.sqlite 'CREATE TABLE nodes (id INTEGER PRIMARY KEY, mass INTEGER NOT NULL, color INTEGER NOT NULL DEFAULT 0);'
	sqlite3 $(DATA)/colors.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO nodes(id, mass) SELECT rowid, mass FROM graph.nodes;'
	DIM=$(DIM) N_NEIGHBORS=$(N_NEIGHBORS) python colors.py $(DATA)

$(DATA)/atlas.sqlite: $(DATA)/graph-umap-$(DIM)-$(N_NEIGHBORS).pkl
	sqlite3 $(DATA)/atlas.sqlite 'CREATE VIRTUAL TABLE nodes USING rtree(id INTEGER PRIMARY KEY, minX INTEGER NOT NULL, maxX INTEGER NOT NULL, minY INTEGER NOT NULL, maxY INTEGER NOT NULL);'
	sqlite3 $(DATA)/atlas.sqlite 'ATTACH DATABASE "$(DATA)/graph.sqlite" AS graph; INSERT INTO nodes(id, minX, maxX, minY, maxY) SELECT rowid, x, x, y, y FROM graph.nodes;'

clean:
	rm -f $(DATA)/graph-*.pkl
	rm -f $(DATA)/atlas.sqlite
	rm -f $(DATA)/colors.sqlite
	rm -f $(DATA)/graph-umap-*.sqlite
