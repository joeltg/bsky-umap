import fs from "node:fs";
const source = fs.readFileSync("./zig-out/bin/wasmtest.wasm");
const typedArray = new Uint8Array(source);

const module = await WebAssembly.instantiate(typedArray, {
  env: {
    print: (result) => {
      console.log(`The result is ${result}`);
    },
    throwError: (ptr, len) => {
      const message = new TextDecoder().decode(
        new Uint8Array(memory.buffer, ptr, len),
      );

      throw new Error(message);
    },
  },
});

const { memory, tile_init, tile_deinit, tile_nodes_ptr } =
  module.instance.exports;

const index = JSON.parse(
  fs.readFileSync(
    "/Users/joelgustafson/Projects/bsky-umap/data/1e5/tiles/index.json",
  ),
  "utf-8",
);

const tile = index.ne.nw;

const atlas = fs.readFileSync(
  `/Users/joelgustafson/Projects/bsky-umap/data/1e5/tiles/${tile.atlas}`,
);

const nodeCount = atlas.byteLength / 16;
console.log("node count", nodeCount);

const tilePtr = tile_init(tile.area.s, tile.area.x, tile.area.y, nodeCount);
console.log("got tilePtr", tilePtr);

const nodePtr = tile_nodes_ptr(tilePtr);
console.log("got nodePtr", nodePtr);
new Uint8Array(memory.buffer, nodePtr, atlas.byteLength).set(atlas);

tile_deinit(tilePtr);
