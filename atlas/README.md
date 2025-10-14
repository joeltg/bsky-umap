# bsky-umap/atlas

## data formats

`*-nodes` files are buffers containing x/y position and color for each node, encoded as 12-byte **little-endian** `[x: f32, y: f32, color: u32]` tuples. colors are rgba8 unormalized; alpha is always 0x00 and is set later by the WebGPU shader.

`*-atlas` files are buffers containing a packed quadtree for each leaf tile. this buffer is also an array of 12-byte nodes in one of two formats for leaf and intermediate nodes.

intermediate nodes hold four **big-endian** u24 links to child nodes (indices into the atlas array). and the first bit is always zero, and a value of 0x000000 represents an empty slot.

leaf nodes are a **big-endian** `[id: u32, x: f32, y: f32]` tuple. the first bit of the id is masked to 1, to distinguish leaf nodes from intermediate nodes. this id is **not** an index into the quadtree array but is the canonical rowid of the user in the SQLite database, and can be used to fetch the user's avatar.

## misc

to hash a `tiles/*` directory:

```
find ${DATA_DIRECTORY}/tiles -type f -exec sha256sum {} \; | sort | sha256sum
```
