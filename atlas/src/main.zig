const std = @import("std");

const cli = @import("zig-cli");
const quadtree = @import("quadtree");

const Atlas = @import("Atlas.zig");
const File = @import("File.zig");

var config = struct {
    path: []const u8 = "",
    capacity: u32 = 80 * 4096, // max 2.6 MB positions, 1.3 MB colors
    dry_run: bool = false,
    list: bool = false,
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
            .long_name = "dry-run",
            .help = "Skip writing node and atlas files",
            .value_ref = r.mkRef(&config.dry_run),
        },
        .{
            .long_name = "list",
            .help = "Write tiles as JSONL stream",
            .value_ref = r.mkRef(&config.list),
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

const JsonStream = std.json.WriteStream(std.fs.File.Writer, .{ .checked_to_fixed_depth = 256 });

const TileWalker = struct {
    allocator: std.mem.Allocator,
    positions: File,
    colors: File,
    ids: File,
    node_count: usize,
    node_indices: []u32,
    atlas: Atlas,
    tile_dir: std.fs.Dir,
    tile_path: std.ArrayList(u8),
    tree: quadtree.Quadtree,

    tile_nodes: std.ArrayList(u8),

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

        const ids_path = try dir.realpath("ids.buffer", &path_buffer);
        const ids = try File.init(ids_path);
        errdefer ids.deinit();

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

        for (0..node_count) |i| {
            node_indices[i] = @intCast(i);
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
            .s = if (s > 0) std.math.pow(f32, 2, @ceil(std.math.log2(s))) else 0,
        };

        std.log.info("got area: [ c = ({d}, {d}), s = {d} ]", .{ area.c[0], area.c[1], area.s });

        return .{
            .allocator = allocator,
            .positions = positions,
            .colors = colors,
            .ids = ids,
            .node_count = node_count,
            .node_indices = node_indices,
            .atlas = Atlas.init(allocator, .{}),
            .tile_dir = tile_dir,
            .tile_path = std.ArrayList(u8).init(allocator),
            .tree = quadtree.Quadtree.init(std.heap.c_allocator, area, .{}),
            .tile_nodes = std.ArrayList(u8).init(allocator),
        };
    }

    pub fn deinit(self: *TileWalker) void {
        self.allocator.free(self.node_indices);
        self.atlas.deinit();
        self.positions.deinit();
        self.colors.deinit();
        self.ids.deinit();
        self.tree.deinit();
        self.tile_path.deinit();
        self.tile_nodes.deinit();
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

        const index_file = switch (config.dry_run) {
            true => std.io.getStdOut(),
            false => try self.tile_dir.createFile("index.json", .{}),
        };

        defer index_file.close();

        self.shuffle(1);
        const area = self.tree.area;

        if (config.list) {
            try self.addTileList(area, 0, index_file.writer());
        } else {
            var stream = std.json.writeStream(index_file.writer(), .{ .whitespace = .indent_tab });
            defer stream.deinit();

            try self.addTile(.{ .c = area.c, .s = area.s }, 0, &stream);
        }
    }

    inline fn shuffle(self: *TileWalker, seed: u64) void {
        var rng = std.Random.Xoshiro256.init(seed);
        rng.random().shuffle(u32, self.node_indices);
    }

    fn addTileList(self: *TileWalker, area: quadtree.Area, idx: u32, out_stream: std.fs.File.Writer) !void {
        const node = self.tree.tree.items[idx];
        const total: u32 = @intFromFloat(@round(node.mass));

        self.tile_nodes.clearRetainingCapacity();

        {
            var stream = std.json.writeStream(out_stream, .{ .whitespace = .minified });
            defer stream.deinit();

            try stream.beginObject();

            try stream.objectField("index");
            try stream.write(idx);
            try stream.objectField("id");
            try stream.write(self.getId());
            try stream.objectField("level");
            try stream.write(self.tile_path.items.len);
            try stream.objectField("total");
            try stream.write(total);

            try stream.objectField("area");
            try stream.beginObject();
            try stream.objectField("x");
            try stream.write(@as(i48, @intFromFloat(area.c[0])));
            try stream.objectField("y");
            try stream.write(@as(i48, @intFromFloat(area.c[1])));
            try stream.objectField("s");
            try stream.write(@as(i48, @intFromFloat(area.s)));
            try stream.endObject();

            // write nodes
            if (!config.dry_run) {
                try stream.objectField("nodes");
                const result = try self.writeFile("nodes", self.tile_nodes.items);
                try result.write(&stream);
            }

            if (total > config.capacity) {
                try stream.objectField("ne");
                try stream.write(if (node.ne != quadtree.Node.NULL) node.ne else null);
                try stream.objectField("nw");
                try stream.write(if (node.nw != quadtree.Node.NULL) node.nw else null);
                try stream.objectField("sw");
                try stream.write(if (node.sw != quadtree.Node.NULL) node.sw else null);
                try stream.objectField("se");
                try stream.write(if (node.se != quadtree.Node.NULL) node.se else null);
            } else {
                // write atlas
                if (!config.dry_run) {
                    const len = self.atlas.tree.items.len * @sizeOf(Atlas.Node);
                    const ptr: [*]const u8 = @ptrCast(self.atlas.tree.items.ptr);
                    try stream.objectField("atlas");
                    const result = try self.writeFile("atlas", ptr[0..len]);
                    try result.write(&stream);
                }
            }

            try stream.endObject();
        }

        try out_stream.writeByte('\n');

        if (total > config.capacity) {
            if (node.ne != quadtree.Node.NULL) {
                try self.tile_path.append('0');
                try self.addTileList(area.divide(.ne), node.ne, out_stream);
                std.debug.assert(self.tile_path.pop() == '0');
            }

            if (node.nw != quadtree.Node.NULL) {
                try self.tile_path.append('1');
                try self.addTileList(area.divide(.nw), node.nw, out_stream);
                std.debug.assert(self.tile_path.pop() == '1');
            }

            if (node.sw != quadtree.Node.NULL) {
                try self.tile_path.append('2');
                try self.addTileList(area.divide(.sw), node.sw, out_stream);
                std.debug.assert(self.tile_path.pop() == '2');
            }

            if (node.se != quadtree.Node.NULL) {
                try self.tile_path.append('3');
                try self.addTileList(area.divide(.se), node.se, out_stream);
                std.debug.assert(self.tile_path.pop() == '3');
            }
        }
    }

    fn addTile(self: *TileWalker, area: Atlas.Area, idx: u32, stream: *JsonStream) !void {
        const node = self.tree.tree.items[idx];
        const total: u32 = @intFromFloat(@round(node.mass));

        self.atlas.reset(area);
        self.tile_nodes.clearRetainingCapacity();

        var count: u32 = 0;
        for (self.node_indices) |i| {
            const id = std.mem.readInt(u32, self.ids.data[i * 4 ..][0..4], .little);
            const x: f32 = @bitCast(std.mem.readInt(u32, self.positions.data[i * 8 ..][0..4], .little));
            const y: f32 = @bitCast(std.mem.readInt(u32, self.positions.data[i * 8 ..][4..8], .little));
            const position = @Vector(2, f32){ x, y };

            if (!area.contains(position)) {
                continue;
            }

            if (!config.dry_run) {
                try self.tile_nodes.appendSlice(self.positions.data[i * 8 ..][0..8]);
                try self.tile_nodes.appendSlice(self.colors.data[i * 4 ..][0..4]);
                try self.atlas.insert(.{ .id = id, .position = position });
            }

            count += 1;
            if (count >= config.capacity) {
                break;
            }
        }

        try stream.beginObject();

        try stream.objectField("id");
        try stream.write(self.getId());
        try stream.objectField("level");
        try stream.write(self.tile_path.items.len);
        try stream.objectField("total");
        try stream.write(total);
        try stream.objectField("count");
        try stream.write(count);

        try stream.objectField("area");
        try stream.beginObject();
        try stream.objectField("x");
        try stream.write(@as(i48, @intFromFloat(area.c[0])));
        try stream.objectField("y");
        try stream.write(@as(i48, @intFromFloat(area.c[1])));
        try stream.objectField("s");
        try stream.write(@as(i48, @intFromFloat(area.s)));
        try stream.endObject();

        // write nodes
        if (!config.dry_run) {
            try stream.objectField("nodes");
            const result = try self.writeFile("nodes", self.tile_nodes.items);
            try result.write(stream);
        }

        if (total > config.capacity) {
            try stream.objectField("ne");
            if (node.ne != quadtree.Node.NULL) {
                try self.tile_path.append('0');
                try self.addTile(area.divide(.ne), node.ne, stream);
                std.debug.assert(self.tile_path.pop() == '0');
            } else try stream.write(null);

            try stream.objectField("nw");
            if (node.nw != quadtree.Node.NULL) {
                try self.tile_path.append('1');
                try self.addTile(area.divide(.nw), node.nw, stream);
                std.debug.assert(self.tile_path.pop() == '1');
            } else try stream.write(null);

            try stream.objectField("sw");
            if (node.sw != quadtree.Node.NULL) {
                try self.tile_path.append('2');
                try self.addTile(area.divide(.sw), node.sw, stream);
                std.debug.assert(self.tile_path.pop() == '2');
            } else try stream.write(null);

            try stream.objectField("se");
            if (node.se != quadtree.Node.NULL) {
                try self.tile_path.append('3');
                try self.addTile(area.divide(.se), node.se, stream);
                std.debug.assert(self.tile_path.pop() == '3');
            } else try stream.write(null);
        } else {
            // write atlas
            if (!config.dry_run) {
                const len = self.atlas.tree.items.len * @sizeOf(Atlas.Node);
                const ptr: [*]const u8 = @ptrCast(self.atlas.tree.items.ptr);
                try stream.objectField("atlas");
                const result = try self.writeFile("atlas", ptr[0..len]);
                try result.write(stream);
            }
        }

        try stream.endObject();
    }

    const WriteFileResult = struct {
        filename: []const u8,
        size: usize,

        pub inline fn write(self: WriteFileResult, stream: *JsonStream) !void {
            try stream.write(self.filename);
        }
    };

    inline fn getId(self: TileWalker) []const u8 {
        return if (self.tile_path.items.len == 0) "root" else self.tile_path.items;
    }

    fn writeFile(self: *TileWalker, name: []const u8, data: []const u8) !WriteFileResult {
        const level = self.tile_path.items.len;
        const id = self.getId();
        const filename = try std.fmt.bufPrint(&path_buffer, "tile-{d}-{s}-{s}", .{ level, id, name });
        try self.tile_dir.writeFile(.{ .sub_path = filename, .data = data });

        try std.io.getStdOut().writer().print("wrote {s} ({d} KB)\n", .{ filename, data.len / 1000 });
        return .{ .filename = filename, .size = data.len };
    }
};
