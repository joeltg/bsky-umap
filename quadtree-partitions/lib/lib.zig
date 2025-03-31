const std = @import("std");
const allocator = std.heap.wasm_allocator;

const Atlas = @import("atlas");

extern fn throwError(ptr: [*]const u8, len: u32) noreturn;

extern fn print(i32) void;
export fn add(a: i32, b: i32) void {
    print(a + b);
}

export fn addFloat(a: f32, b: f32) f32 {
    return (a + b);
}

pub fn panic(message: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    throwError(message.ptr, message.len);
}

const Tile = struct {
    area: Atlas.Area,
    nodes: []Atlas.Node,
};

var tiles = std.ArrayList(Tile).init(allocator);

export fn tile_init(s: f32, c_x: f32, c_y: f32, node_count: u32) ?*const Tile {
    const tile = allocator.create(Tile) catch
        @panic("failed to allocate tile");
    tile.area.s = s;
    tile.area.c[0] = c_x;
    tile.area.c[1] = c_y;
    tile.nodes = allocator.alloc(Atlas.Node, node_count) catch
        @panic("failed to allocate nodes");
    return tile;
}

export fn tile_deinit(tile: *const Tile) void {
    allocator.free(tile.nodes);
    allocator.destroy(tile);
}

export fn tile_nodes_len(tile: *const Tile) usize {
    return tile.nodes.len;
}

export fn tile_nodes_ptr(tile: *const Tile) [*]Atlas.Node {
    return tile.nodes.ptr;
}

var nearest_body: Atlas.Body = undefined;

export fn get_nearest_body_inclusive(tile: *const Tile, x: f32, y: f32) *Atlas.Body {
    nearest_body = Atlas.getNearestBody(tile.nodes, tile.area, .{ x, y }, .inclusive) catch
        @panic("failed to get nearest body");
    return &nearest_body;
}

export fn get_nearest_body_exclusive(tile: *const Tile, x: f32, y: f32) *Atlas.Body {
    nearest_body = Atlas.getNearestBody(tile.nodes, tile.area, .{ x, y }, .exclusive) catch
        @panic("failed to get nearest body");
    return &nearest_body;
}

// const Iterator = struct {
//     min_x: f32,
//     max_x: f32,
//     min_y: f32,
//     max_y: f32,

//     pub fn init(min_x: f32, max_x: f32, min_y: f32, max_y: f32) Iterator {
//         return .{
//             .min_x = min_x,
//             .max_x = max_x,
//             .min_y = min_y,
//             .max_y = max_y,
//         };
//     }

//     pub fn next(self: Iterator) !?*Atlas.Body {
//         return null;
//     }
// };

// export fn get_node_x(ptr: *Tile) void {}
