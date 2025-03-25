const std = @import("std");

const cli = @import("zig-cli");
const quadtree = @import("quadtree");

const File = @import("File.zig");

var config = struct {
    path: []const u8 = "",
    capacity: u32 = 80 * 4096, // max 2.6 MB positions, 1.3 MB colors
    write: bool = false,
}{};

pub fn main() !void {
    var r = try cli.AppRunner.init(std.heap.page_allocator);

    const args: []const cli.PositionalArg = &.{.{
        .name = "path",
        .help = "path to data directory",
        .value_ref = r.mkRef(&config.path),
    }};

    const options: []const cli.Option = &.{
        .{
            .long_name = "capacity",
            .short_alias = 'c',
            .help = "Maximumm capacity of each tile",
            .value_ref = r.mkRef(&config.capacity),
        },
        .{
            .long_name = "write",
            .short_alias = 'w',
            .help = "Write tiles to tiles/ folder",
            .value_ref = r.mkRef(&config.write),
        },
    };

    const app = &cli.App{
        .command = .{
            .name = "quadtree-partitions",
            .target = cli.CommandTarget{
                .action = cli.CommandAction{
                    .exec = run,
                    .positional_args = .{ .required = args },
                },
            },
            .options = options,
        },
    };

    return r.run(app);
}

var path_buffer: [std.fs.max_path_bytes]u8 = undefined;

fn run() !void {
    std.log.info("opening directory {s}", .{config.path});
    var dir = try std.fs.cwd().openDir(config.path, .{});
    defer dir.close();

    var walker = try TileWalker.init(std.heap.c_allocator, &dir);
    defer walker.deinit();

    try walker.build();
}

const Tile = struct {
    level: u32,
    count: u32,
    area: quadtree.Area,
};

const TileWalker = struct {
    allocator: std.mem.Allocator,
    positions: File,
    colors: File,
    node_count: usize,
    node_indices: []u32,
    tile_dir: std.fs.Dir,
    tile_path: std.ArrayList(u8),
    tiles: std.ArrayList(Tile),
    tree: quadtree.Quadtree,

    tile_positions: std.ArrayList(u8),
    tile_colors: std.ArrayList(u8),
    rng: std.Random.Xoshiro256,

    pub fn init(allocator: std.mem.Allocator, dir: *std.fs.Dir) !TileWalker {
        try dir.deleteTree("tiles");
        try dir.makeDir("tiles");
        const tile_dir = try dir.openDir("tiles", .{});
        const positions_path = try dir.realpath("positions.buffer", &path_buffer);
        const positions = try File.init(positions_path);
        errdefer positions.deinit();

        const colors_path = try dir.realpath("colors.buffer", &path_buffer);
        const colors = try File.init(colors_path);
        errdefer colors.deinit();

        std.log.info("positions.len: {d}", .{positions.data.len});
        std.log.info("colors.len: {d}", .{colors.data.len});
        std.debug.assert(positions.data.len == colors.data.len * 2);

        const node_count = positions.data.len / 8;
        const node_indices = try allocator.alloc(u32, node_count);
        errdefer allocator.free(node_indices);

        var min_x: f32 = 0;
        var max_x: f32 = 0;
        var min_y: f32 = 0;
        var max_y: f32 = 0;

        for (node_indices, 0..) |*index, i| {
            index.* = @intCast(i);
            const offset = i * 8;
            const x: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(positions.data[offset .. offset + 4]), .little));
            const y: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(positions.data[offset + 4 .. offset + 8]), .little));
            min_x = @min(min_x, x);
            max_x = @max(max_x, x);
            min_y = @min(min_y, y);
            max_y = @max(max_y, y);
        }

        const s = 2 * @max(@abs(min_x), @abs(max_x), @abs(min_y), @abs(max_y));
        const area = quadtree.Area{
            .c = .{ 0, 0 },
            // .s = s,
            .s = if (s > 0) std.math.pow(f32, 2, @ceil(std.math.log2(s))) else 0,
        };

        std.log.info("got area: [ c = ({d}, {d}), s = {d} ]", .{ area.c[0], area.c[1], area.s });

        return .{
            .allocator = allocator,
            .positions = positions,
            .colors = colors,
            .node_count = node_count,
            .node_indices = node_indices,
            .tile_dir = tile_dir,
            .tile_path = std.ArrayList(u8).init(allocator),
            .tiles = std.ArrayList(Tile).init(allocator),
            .tree = quadtree.Quadtree.init(std.heap.c_allocator, area, .{}),
            .tile_positions = std.ArrayList(u8).init(allocator),
            .tile_colors = std.ArrayList(u8).init(allocator),
            .rng = std.Random.Xoshiro256.init(0),
        };
    }

    pub fn deinit(self: *TileWalker) void {
        self.allocator.free(self.node_indices);
        self.positions.deinit();
        self.colors.deinit();
        self.tree.deinit();
        self.tiles.deinit();
        self.tile_path.deinit();
        self.tile_positions.deinit();
        self.tile_colors.deinit();
        self.tile_dir.close();
    }

    pub fn build(self: *TileWalker) !void {
        for (0..self.node_count) |i| {
            const offset = i * 8;
            const x: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(self.positions.data[offset .. offset + 4]), .little));
            const y: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(self.positions.data[offset + 4 .. offset + 8]), .little));
            try self.tree.insert(.{ x, y }, 1.0);
        }

        std.log.info("inserted {d} points into quadtree", .{self.node_count});

        try self.addTile(self.tree.area, 0);
        std.log.info("TILES", .{});
        var sum: u32 = 0;
        for (self.tiles.items) |tile| {
            std.log.info(
                "- level {d}, count {d}, area [ c = ({d}, {d}), s = {d} ]",
                .{ tile.level, tile.count, tile.area.c[0], tile.area.c[1], tile.area.s },
            );
            sum += @min(tile.count, config.capacity);
        }

        std.log.info("got {d} tiles ({d} total count)", .{ self.tiles.items.len, sum });
    }

    inline fn shuffle(self: *TileWalker) void {
        self.rng.random().shuffle(u32, self.node_indices);
    }

    fn addTile(self: *TileWalker, area: quadtree.Area, id: u32) !void {
        const node = self.tree.tree.items[id];
        const count: u32 = @intFromFloat(@round(node.mass));
        try self.tiles.append(.{ .level = @intCast(self.tile_path.items.len), .count = count, .area = area });

        if (config.write) {
            self.tile_positions.clearRetainingCapacity();
            self.tile_colors.clearRetainingCapacity();
            self.shuffle();

            for (self.node_indices) |i| {
                const x: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(self.positions.data[i * 8 .. i * 8 + 4]), .little));
                const y: f32 = @bitCast(std.mem.readInt(u32, @ptrCast(self.positions.data[i * 8 + 4 .. i * 8 + 8]), .little));
                if (area.contains(.{ x, y })) {
                    try self.tile_colors.appendSlice(self.colors.data[i * 4 .. i * 4 + 4]);
                    try self.tile_positions.appendSlice(self.positions.data[i * 8 .. i * 8 + 8]);

                    if (self.tile_colors.items.len >= config.capacity) {
                        break;
                    }
                }
            }

            try self.writeFile("positions.buffer", self.tile_positions.items);
            try self.writeFile("colors.buffer", self.tile_colors.items);
        }

        if (count > config.capacity) {
            if (node.ne != quadtree.Node.NULL) {
                try self.tile_path.append('0');
                try self.addTile(area.divide(.ne), node.ne);
                std.debug.assert(self.tile_path.pop() == '0');
            }

            if (node.nw != quadtree.Node.NULL) {
                try self.tile_path.append('1');
                try self.addTile(area.divide(.nw), node.nw);
                std.debug.assert(self.tile_path.pop() == '1');
            }

            if (node.sw != quadtree.Node.NULL) {
                try self.tile_path.append('2');
                try self.addTile(area.divide(.sw), node.sw);
                std.debug.assert(self.tile_path.pop() == '2');
            }

            if (node.se != quadtree.Node.NULL) {
                try self.tile_path.append('3');
                try self.addTile(area.divide(.se), node.se);
                std.debug.assert(self.tile_path.pop() == '3');
            }
        }
    }

    fn writeFile(self: *TileWalker, name: []const u8, data: []const u8) !void {
        const level = self.tile_path.items.len;
        const path = if (level == 0) "root" else self.tile_path.items;
        const filename = try std.fmt.bufPrint(&path_buffer, "tile-{d}-{s}-{s}", .{ level, path, name });
        try self.tile_dir.writeFile(.{ .sub_path = filename, .data = data });

        const stdout = std.io.getStdOut();
        try stdout.writer().print("wrote {s} ({d} KB)\n", .{ filename, data.len / 1000 });
    }
};
