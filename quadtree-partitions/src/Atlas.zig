const std = @import("std");

const Atlas = @This();

pub const Quadrant = enum(u2) {
    sw = 0,
    nw = 1,
    se = 2,
    ne = 3,
};

pub const Area = packed struct {
    s: f32 = 0,
    c: @Vector(2, f32) = .{ 0, 0 },

    pub fn locate(area: Area, point: @Vector(2, f32)) Quadrant {
        const q = point < area.c;

        if (q[0]) {
            if (q[1]) {
                return .sw;
            } else {
                return .nw;
            }
        } else {
            if (q[1]) {
                return .se;
            } else {
                return .ne;
            }
        }
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
    id: u32 = Node.NULL,
    position: @Vector(2, f32) = .{ 0, 0 },
};

/// We can pack both nodes and leaves into 16 bytes total,
/// while still distinguishing between them.
///
/// Intermediate nodes hold up to four links,
/// and use Node.NULL for empty slots.
///
/// Leaf nodes have { id, x, y, 0 }.
/// The last slot node.se = 0 distinguishes leafs from nodes,
/// since nodes can never link to index zero.
pub const Node = packed struct {
    pub const NULL = std.math.maxInt(u32);

    ne: u32 = NULL,
    nw: u32 = NULL,
    sw: u32 = NULL,
    se: u32 = NULL,

    pub fn create(body: Body) Node {
        var node = Node{};
        node.setLeaf(body);
        return node;
    }

    pub inline fn isLeaf(node: Node) bool {
        return node.se == 0;
    }

    pub inline fn clear(self: *Node) void {
        self.ne = NULL;
        self.nw = NULL;
        self.sw = NULL;
        self.se = NULL;
    }

    pub inline fn setLeaf(self: *Node, body: Body) void {
        self.ne = body.id;
        self.nw = @bitCast(body.position[0]);
        self.sw = @bitCast(body.position[1]);
        self.se = 0;
    }

    pub inline fn getId(self: Node) u32 {
        std.debug.assert(self.se == 0);
        return self.ne;
    }

    pub inline fn getPosition(self: Node) @Vector(2, f32) {
        std.debug.assert(self.se == 0);
        return .{ @bitCast(self.nw), @bitCast(self.sw) };
    }

    pub inline fn getQuadrant(node: Node, quadrant: Quadrant) u32 {
        return switch (quadrant) {
            .ne => node.ne,
            .nw => node.nw,
            .sw => node.sw,
            .se => node.se,
        };
    }

    pub inline fn setQuadrant(node: *Node, quadrant: Quadrant, index: u32) void {
        switch (quadrant) {
            .ne => node.ne = index,
            .nw => node.nw = index,
            .sw => node.sw = index,
            .se => node.se = index,
        }
    }
};

pub const Error = std.mem.Allocator.Error || error{ Empty, OutOfBounds };

area: Area,
tree: std.ArrayList(Node),

pub fn init(allocator: std.mem.Allocator, area: Area) Atlas {
    return .{
        .tree = std.ArrayList(Node).init(allocator),
        .area = area,
    };
}

pub fn deinit(self: Atlas) void {
    self.tree.deinit();
}

pub fn reset(self: *Atlas, area: Area) void {
    self.area = area;
    self.tree.clearRetainingCapacity();
}

pub fn insert(self: *Atlas, body: Body) !void {
    if (!self.area.contains(body.position))
        return Error.OutOfBounds;

    if (self.tree.items.len == 0) {
        try self.tree.append(Node.create(body));
    } else {
        try self.insertNode(0, self.area, body);
    }
}

fn insertNode(self: *Atlas, id: u32, area: Area, body: Body) !void {
    std.debug.assert(id < self.tree.items.len);
    std.debug.assert(area.s > 0);

    if (self.tree.items[id].isLeaf()) {
        const node = self.tree.items[id];
        const node_position = node.getPosition();
        const node_quadrant = area.locate(node_position);

        if (@reduce(.And, node_position == body.position))
            return error.Collision;

        const index: u32 = @intCast(self.tree.items.len);
        try self.tree.append(node);

        self.tree.items[id].clear();
        self.tree.items[id].setQuadrant(node_quadrant, index);
    }

    const body_quadrant = area.locate(body.position);
    const child = self.tree.items[id].getQuadrant(body_quadrant);

    if (child != Node.NULL) {
        try self.insertNode(child, area.divide(body_quadrant), body);
    } else {
        const index: u32 = @intCast(self.tree.items.len);
        try self.tree.append(Node.create(body));
        self.tree.items[id].setQuadrant(body_quadrant, index);
    }
}

pub const NearestBodyMode = enum(u2) { inclusive, exclusive };

pub fn getNearestBody(self: Atlas, position: @Vector(2, f32), mode: NearestBodyMode) !Body {
    if (self.tree.items.len == 0)
        return error.Empty;

    var nearest = Body{};
    var neartest_dist = std.math.inf(f32);
    self.getNearestBodyNode(0, self.area, position, mode, &nearest, &neartest_dist);
    return nearest;
}

fn getNearestBodyNode(
    self: Atlas,
    id: u32,
    area: Area,
    position: @Vector(2, f32),
    mode: NearestBodyMode,
    nearest: *Body,
    nearest_dist: *f32,
) void {
    if (id >= self.tree.items.len)
        @panic("index out of range");

    const node = self.tree.items[id];

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
        if (node.sw != Node.NULL)
            self.getNearestBodyNode(node.sw, area.divide(.sw), position, mode, nearest, nearest_dist);
        if (node.nw != Node.NULL)
            self.getNearestBodyNode(node.nw, area.divide(.nw), position, mode, nearest, nearest_dist);
        if (node.se != Node.NULL)
            self.getNearestBodyNode(node.se, area.divide(.se), position, mode, nearest, nearest_dist);
        if (node.ne != Node.NULL)
            self.getNearestBodyNode(node.ne, area.divide(.ne), position, mode, nearest, nearest_dist);
    }
}

pub fn print(self: *Atlas, log: std.fs.File.Writer) !void {
    try self.printNode(log, 0, 1);
}

fn printNode(self: *Atlas, log: std.fs.File.Writer, id: u32, depth: usize) !void {
    if (id >= self.tree.items.len)
        @panic("index out of range");

    const node = self.tree.items[id];
    if (node.isLeaf()) {
        try log.print("leaf {d} - {d}\n", .{ id, node.getId() });
        return;
    }

    try log.print("node {d}\n", .{id});
    if (node.sw != Node.NULL) {
        try log.writeByteNTimes(' ', depth * 2);
        try log.print("sw: ", .{});
        try self.printNode(log, node.sw, depth + 1);
    }
    if (node.nw != Node.NULL) {
        try log.writeByteNTimes(' ', depth * 2);
        try log.print("nw: ", .{});
        try self.printNode(log, node.nw, depth + 1);
    }
    if (node.se != Node.NULL) {
        try log.writeByteNTimes(' ', depth * 2);
        try log.print("se: ", .{});
        try self.printNode(log, node.se, depth + 1);
    }
    if (node.ne != Node.NULL) {
        try log.writeByteNTimes(' ', depth * 2);
        try log.print("ne: ", .{});
        try self.printNode(log, node.ne, depth + 1);
    }
}

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
