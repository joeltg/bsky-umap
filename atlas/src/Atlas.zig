const std = @import("std");

const Atlas = @This();

pub const Quadrant = enum(u2) {
    ne,
    nw,
    sw,
    se,
};

pub const Area = packed struct {
    s: f32 = 0,
    c: @Vector(2, f32) = .{ 0, 0 },

    pub fn locate(area: Area, point: @Vector(2, f32)) Quadrant {
        const x, const y = point < area.c;
        return switch (@as(u2, @intFromBool(y)) << 1 | @intFromBool(x)) {
            0b00 => .ne,
            0b01 => .nw,
            0b10 => .se,
            0b11 => .sw,
        };
    }

    pub fn divide(area: Area, quadrant: Quadrant) Area {
        const s = area.s / 2;

        var delta: @Vector(2, f32) = switch (quadrant) {
            .sw => .{ -1, -1 },
            .nw => .{ -1, 1 },
            .se => .{ 1, -1 },
            .ne => .{ 1, 1 },
        };

        delta *= @splat(s / 2);

        return .{ .s = s, .c = area.c + delta };
    }

    pub fn contains(area: Area, point: @Vector(2, f32)) bool {
        const s = area.s / 2;
        const min_x = area.c[0] - s;
        const max_x = area.c[0] + s;
        const min_y = area.c[1] - s;
        const max_y = area.c[1] + s;
        if (point[0] < min_x or max_x < point[0]) return false;
        if (point[1] < min_y or max_y < point[1]) return false;
        return true;
    }

    pub fn getMinDistance(area: Area, point: @Vector(2, f32)) f32 {
        const zero: @Vector(2, f32) = comptime @splat(0);
        const s: @Vector(2, f32) = @splat(area.s / 2);
        const d = @abs(point - area.c) - s;
        return getNorm(2, @max(d, zero));
    }
};

pub const Body = packed struct {
    id: u32 = 0,
    position: @Vector(2, f32) = .{ 0, 0 },
};

/// We can pack both nodes and leaves into 12 bytes total,
/// while still distinguishing between them.
///
/// Intermediate nodes hold up to four u24 links,
/// each with a leading bit set to 0, use 0 for empty slots.
///
/// Leaves have { id, x, y } of 4 bytes each. the id is always
/// flagged with 0x80000000 so it can be distinguished from nodes.
pub const Node = struct {

    // zig-fmt: off
    data: [12]u8 = .{
        0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff,
    },
    // zig-fmt: on

    const max_id = (1 << 31) ^ std.math.maxInt(u32);
    pub fn create(body: Body) Node {
        std.debug.assert(body.id <= max_id);

        var node = Node{};

        const flag: u32 = comptime (1 << 31);
        std.mem.writeInt(u32, node.data[0..4], flag | body.id, .big);
        std.mem.writeInt(u32, node.data[4..8], @bitCast(body.position[0]), .big);
        std.mem.writeInt(u32, node.data[8..12], @bitCast(body.position[1]), .big);

        return node;
    }

    pub inline fn isLeaf(self: Node) bool {
        return 0x80 & self.data[0] != 0;
    }

    pub inline fn clear(self: *Node) void {
        self.setQuadrant(.ne, 0);
        self.setQuadrant(.nw, 0);
        self.setQuadrant(.sw, 0);
        self.setQuadrant(.se, 0);
    }

    pub inline fn getId(self: Node) u32 {
        const flag: u32 = comptime (1 << 31);
        const mask = std.math.maxInt(u32) ^ flag;
        return mask & std.mem.readInt(u32, self.data[0..4], .big);
    }

    pub inline fn getPosition(self: Node) @Vector(2, f32) {
        return .{
            @bitCast(std.mem.readInt(u32, self.data[4..8], .big)),
            @bitCast(std.mem.readInt(u32, self.data[8..12], .big)),
        };
    }

    pub inline fn getQuadrant(node: Node, quadrant: Quadrant) u24 {
        return switch (quadrant) {
            .ne => std.mem.readInt(u24, node.data[0..3], .big),
            .nw => std.mem.readInt(u24, node.data[3..6], .big),
            .sw => std.mem.readInt(u24, node.data[6..9], .big),
            .se => std.mem.readInt(u24, node.data[9..12], .big),
        };
    }

    pub inline fn setQuadrant(node: *Node, quadrant: Quadrant, value: u24) void {
        switch (quadrant) {
            .ne => std.mem.writeInt(u24, node.data[0..3], value, .big),
            .nw => std.mem.writeInt(u24, node.data[3..6], value, .big),
            .sw => std.mem.writeInt(u24, node.data[6..9], value, .big),
            .se => std.mem.writeInt(u24, node.data[9..12], value, .big),
        }
    }
};

comptime {
    if (@sizeOf(Node) != 12) {
        @compileError("expected @sizeOf(Node) == 12");
    }
}

pub const Error = std.mem.Allocator.Error || error{ Empty, OutOfBounds };

allocator: std.mem.Allocator,
area: Area,
tree: std.ArrayList(Node),
count: std.ArrayList(usize),

pub fn init(allocator: std.mem.Allocator, area: Area) Atlas {
    return .{
        .allocator = allocator,
        .area = area,
        .tree = std.ArrayList(Node).empty,
        .count = std.ArrayList(usize).empty,
    };
}

pub fn deinit(self: *Atlas) void {
    self.tree.deinit(self.allocator);
    self.count.deinit(self.allocator);
}

pub fn reset(self: *Atlas, area: Area) void {
    self.area = area;
    self.tree.clearRetainingCapacity();
    self.count.clearRetainingCapacity();
}

pub fn insert(self: *Atlas, body: Body) !void {
    if (!self.area.contains(body.position))
        return Error.OutOfBounds;

    if (self.tree.items.len == 0) {
        _ = try self.append(Node.create(body));
    } else {
        try self.insertNode(0, self.area, body);
    }
}

fn insertNode(self: *Atlas, idx: u24, area: Area, body: Body) !void {
    std.debug.assert(idx < self.tree.items.len);
    std.debug.assert(area.s > 0);

    if (self.tree.items[idx].isLeaf()) {
        const node = self.tree.items[idx];
        const node_position = node.getPosition();
        const node_quadrant = area.locate(node_position);

        if (@reduce(.And, node_position == body.position)) {
            std.log.err(
                "COLLISION: existing node {d} at ({d}, {d}) while inserting {d} at ({d}, {d})",
                .{ node.getId(), node_position[0], node_position[1], body.id, body.position[0], body.position[1] },
            );

            return error.Collision;
        }

        const child_idx = try self.append(node);
        self.tree.items[idx].clear();
        self.tree.items[idx].setQuadrant(node_quadrant, child_idx);
    }

    self.count.items[idx] += 1;

    const body_quadrant = area.locate(body.position);
    const child = self.tree.items[idx].getQuadrant(body_quadrant);

    if (child != 0) {
        try self.insertNode(child, area.divide(body_quadrant), body);
    } else {
        const index = try self.append(Node.create(body));
        self.tree.items[idx].setQuadrant(body_quadrant, index);
    }
}

inline fn append(self: *Atlas, node: Node) !u24 {
    const index: u24 = @intCast(self.tree.items.len);
    try self.tree.append(self.allocator, node);
    try self.count.append(self.allocator, 1);
    return index;
}

pub const NearestBodyMode = enum(u2) { inclusive, exclusive };

pub fn getNearestBody(nodes: []const Node, area: Area, position: @Vector(2, f32), mode: NearestBodyMode) !Body {
    if (nodes.len == 0)
        return error.Empty;

    var nearest = Body{};
    var neartest_dist = std.math.inf(f32);
    getNearestBodyNode(nodes, area, 0, position, mode, &nearest, &neartest_dist);
    return nearest;
}

fn getNearestBodyNode(
    nodes: []const Node,
    area: Area,
    idx: u32,
    position: @Vector(2, f32),
    mode: NearestBodyMode,
    nearest: *Body,
    nearest_dist: *f32,
) void {
    const node = nodes[idx];
    if (node.isLeaf()) {
        const node_position = node.getPosition();
        if (@reduce(.And, node_position == position) and mode == .exclusive)
            return;

        const dist = getNorm(2, node_position - position);
        if (dist < nearest_dist.*) {
            nearest.id = node.getId();
            nearest.position = node_position;
            nearest_dist.* = dist;
        }
    } else if (area.getMinDistance(position) < nearest_dist.*) {
        switch (node.getQuadrant(.sw)) {
            0 => {},
            else => |child| getNearestBodyNode(nodes, area.divide(.sw), child, position, mode, nearest, nearest_dist),
        }
        switch (node.getQuadrant(.nw)) {
            0 => {},
            else => |child| getNearestBodyNode(nodes, area.divide(.nw), child, position, mode, nearest, nearest_dist),
        }
        switch (node.getQuadrant(.se)) {
            0 => {},
            else => |child| getNearestBodyNode(nodes, area.divide(.se), child, position, mode, nearest, nearest_dist),
        }
        switch (node.getQuadrant(.ne)) {
            0 => {},
            else => |child| getNearestBodyNode(nodes, area.divide(.ne), child, position, mode, nearest, nearest_dist),
        }
    }
}

// pub fn print(self: *Atlas, log: std.fs.File.Writer) !void {
//     try self.printNode(log, 0, 1);
// }

// fn printNode(self: *Atlas, log: std.fs.File.Writer, idx: u32, depth: usize) !void {
//     const node = self.tree.items[idx];
//     if (node.isLeaf()) {
//         try log.print("leaf {d} - {d}\n", .{ idx, node.getId() });
//         return;
//     }

//     try log.print("node {d}\n", .{idx});
//     if (node.sw != Node.NULL) {
//         try log.writeByteNTimes(' ', depth * 2);
//         try log.print("sw: ", .{});
//         try self.printNode(log, node.sw, depth + 1);
//     }
//     if (node.nw != Node.NULL) {
//         try log.writeByteNTimes(' ', depth * 2);
//         try log.print("nw: ", .{});
//         try self.printNode(log, node.nw, depth + 1);
//     }
//     if (node.se != Node.NULL) {
//         try log.writeByteNTimes(' ', depth * 2);
//         try log.print("se: ", .{});
//         try self.printNode(log, node.se, depth + 1);
//     }
//     if (node.ne != Node.NULL) {
//         try log.writeByteNTimes(' ', depth * 2);
//         try log.print("ne: ", .{});
//         try self.printNode(log, node.ne, depth + 1);
//     }
// }

pub fn getNorm(comptime R: u3, f: @Vector(R, f32)) f32 {
    return std.math.sqrt(@reduce(.Add, f * f));
}

test "create and construct Atlas" {
    const s: f32 = 256000;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var prng = std.Random.Xoshiro256.init(0);
    const random = prng.random();

    var atlas = Atlas.init(allocator, .{ .s = s });
    defer atlas.deinit();

    // var bodies = std.ArrayList(Body).init(allocator);
    // defer bodies.deinit();

    const count = 100;
    for (0..count) |_| {
        const x = (random.float(f32) - 0.5) * s;
        const y = (random.float(f32) - 0.5) * s;
        const id: u32 = random.uintLessThan(u32, std.math.maxInt(u32));

        try atlas.insert(.{ .id = id, .position = .{ x, y } });
        // try bodies.append(.{ .id = id, .position = .{ x, y } });
    }

    const stdout = std.io.getStdOut();
    try atlas.print(stdout.writer());

    try stdout.writer().print("@sizeOf(Node): {d}\n", .{@sizeOf(Node)});

    // std.log.warn("internal nodes: {d}", .{atlas.tree.items.len});
    // std.log.warn("total size: {d}", .{atlas.tree.items.len * @sizeOf(Node)});
    // std.log.warn("baseline size: {d}", .{count * @sizeOf(Body)});
}

test "getNearestBody" {
    const s: f32 = 256;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer std.testing.expect(gpa.deinit() == .ok) catch {};

    var qt = Atlas.init(allocator, .{ .s = s });
    defer qt.deinit();

    // Test empty tree
    try std.testing.expectError(error.Empty, qt.getNearestBody(.{ 0, 0 }, .inclusive));

    // Insert bodies in different quadrants
    const p1: @Vector(2, f32) = .{ 10, 10 };
    const p2: @Vector(2, f32) = .{ 100, 100 };
    const p3: @Vector(2, f32) = .{ -50, -50 };

    try qt.insert(.{ .id = 1, .position = p1 });
    try qt.insert(.{ .id = 2, .position = p2 });
    try qt.insert(.{ .id = 3, .position = p3 });

    { // Test finding nearest to a point
        const nearest = try qt.getNearestBody(.{ 15, 15 }, .inclusive);

        // p1 should be nearest to the query point
        try std.testing.expectEqual(1, nearest.id);
        try std.testing.expect(@reduce(.And, nearest.position == p1));
    }

    { // Test finding nearest to a child, inclusive
        const nearest = try qt.getNearestBody(.{ 10, 10 }, .inclusive);

        // p1 should be nearest to the query point
        try std.testing.expectEqual(1, nearest.id);
        try std.testing.expect(@reduce(.And, nearest.position == p1));
    }

    { // Test finding nearest to a child, exclusive
        const nearest = try qt.getNearestBody(.{ 10, 10 }, .exclusive);

        // p1 should be nearest to the query point
        try std.testing.expectEqual(3, nearest.id);
        try std.testing.expect(@reduce(.And, nearest.position == p3));
    }
}
