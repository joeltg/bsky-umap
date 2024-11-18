source .env

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path/to/dir"
    exit 1
fi

if [ -z "${DIM+x}" ]; then
    echo "Error: DIM is not defined."
    exit 1
fi

if [ -z "${N_NEIGHBORS+x}" ]; then
    echo "Error: N_NEIGHBORS is not defined."
    exit 1
fi

DB=${1}/db.sqlite
ATLAS=${1}/atlas-umap-${DIM}-${N_NEIGHBORS}.sqlite
GRAPH=${1}/graph-umap-${DIM}-${N_NEIGHBORS}.sqlite

echo "DB=${DB}"
echo "ATLAS=${ATLAS}"
echo "GRAPH=${GRAPH}"

sqlite3 <<EOF
    ATTACH DATABASE '${DB}' AS db;
    ATTACH DATABASE '${ATLAS}' as atlas;
    ATTACH DATABASE '${GRAPH}' as graph;

    UPDATE graph.nodes SET mass = db.users.incoming_follow_count FROM db.users WHERE db.users.id = graph.nodes.id;

    DROP TABLE IF EXISTS atlas.nodes;
    CREATE TABLE atlas.nodes (id INTEGER PRIMARY KEY, mass INTEGER NOT NULL, color INTEGER NOT NULL);

    INSERT INTO atlas.nodes(id, mass, color) SELECT id, mass, color FROM graph.nodes;

    DROP TABLE IF EXISTS atlas.users;
    CREATE VIRTUAL TABLE atlas.users USING rtree(
        id INTEGER PRIMARY KEY,
        minX INTEGER NOT NULL,
        maxX INTEGER NOT NULL,
        minY INTEGER NOT NULL,
        maxY INTEGER NOT NULL
    );

    INSERT INTO atlas.users(id, minX, maxX, minY, maxY) SELECT id, x, x, y, y FROM graph.nodes;
EOF
