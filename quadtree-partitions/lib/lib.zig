const std = @import("std");
const allocator = std.heap.page_allocator;

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
    nodes: []const Atlas.Node,
};

var tiles = std.ArrayList(Tile).init(allocator);

// export fn init(s: i32, x: i32, y: i32, node_count: u32) ?[*]Atlas.Node {
export fn init(node_count: u32) ?[*]Atlas.Node {
    // const area = Atlas.Area{
    //     .s = @floatFromInt(s),
    //     .c = .{ @floatFromInt(x), @floatFromInt(y) },
    // };

    // tiles.append(.{ .area = area, .nodes = nodes }) catch return null;
    const nodes = allocator.alloc(Atlas.Node, node_count) catch
        @panic("failed to allocate nodes");
    return nodes.ptr;
}

var nearest_body: Atlas.Body = undefined;

export fn get_nearest_body_inclusive(
    nodes_ptr: [*]const Atlas.Node,
    nodes_len: usize,
    area_s: f32,
    area_x: f32,
    area_y: f32,
    x: f32,
    y: f32,
) *Atlas.Body {
    const nodes = nodes_ptr[0..nodes_len];
    const area = Atlas.Area{ .s = area_s, .c = .{ area_x, area_y } };
    nearest_body = Atlas.getNearestBody(nodes, area, .{ x, y }, .inclusive) catch
        @panic("failed to get nearest body");
    return &nearest_body;
}

export fn get_nearest_body_exclusive(
    nodes_ptr: [*]const Atlas.Node,
    nodes_len: usize,
    area_s: f32,
    area_x: f32,
    area_y: f32,
    x: f32,
    y: f32,
) *Atlas.Body {
    const nodes = nodes_ptr[0..nodes_len];
    const area = Atlas.Area{ .s = area_s, .c = .{ area_x, area_y } };
    nearest_body = Atlas.getNearestBody(nodes, area, .{ x, y }, .exclusive) catch
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
