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
    echo "Error: N_NEIGHBORSS is not defined."
    exit 1
fi

sqlite3 <<EOF
    ATTACH DATABASE '${1}/db.sqlite' AS db;
    ATTACH DATABASE '${1}/atlas-umap-${DIM}-${N_NEIGHBORS}.sqlite' as atlas;
    ATTACH DATABASE '${1}/graph-umap-${DIM}-${N_NEIGHBORS}.sqlite' as graph;

    UPDATE graph.nodes SET mass = db.users.incoming_follow_count FROM db.users WHERE db.users.id = graph.nodes.rowid;

    CREATE VIRTUAL TABLE IF NOT EXISTS atlas.users USING rtree(
        id INTEGER PRIMARY KEY,
        minX INTEGER NOT NULL,
        maxX INTEGER NOT NULL,
        minY INTEGER NOT NULL,
        maxY INTEGER NOT NULL,
        minZ INTEGER NOT NULL,
        maxZ INTEGER NOT NULL
    );

    DELETE FROM atlas.users;

    INSERT INTO atlas.users(id, minX, maxX, minY, maxY, minZ, maxZ) SELECT rowid, x, x, y, y, mass, mass FROM graph.nodes;
EOF
