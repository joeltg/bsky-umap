const std = @import("std");

const cli = @import("zig-cli");
const quadtree = @import("quadtree");
const sqlite = @import("sqlite");

var config = struct {
    path: []const u8 = "",
    capacity: u32 = 10000,
}{};

pub fn main() !void {
    var r = try cli.AppRunner.init(std.heap.page_allocator);

    const args: []const cli.PositionalArg = &.{.{
        .name = "path",
        .help = "path to data directory",
        .value_ref = r.mkRef(&config.path),
    }};

    const options: []const cli.Option = &.{.{
        .long_name = "capacity",
        .short_alias = 'c',
        .help = "Maximumm capacity of each tile",
        .value_ref = r.mkRef(&config.capacity),
    }};

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

fn run() !void {
    var path: [std.fs.max_path_bytes]u8 = undefined;
    @memcpy(path[0..config.path.len], config.path);
    path[config.path.len] = 0;

    std.log.info("opening {s}", .{path[0..config.path.len :0]});

    const db = try sqlite.Database.open(.{
        .path = path[0..config.path.len :0],
        .create = false,
    });
    defer db.close();

    const area = try getArea(db);

    std.log.info("GOT AREA [ c = ({d}, {d}), s = {d} ]", .{ area.c[0], area.c[1], area.s });

    const SelectNodesParams = struct {};
    const SelectNodesResult = struct { x: f32, y: f32 };
    const select_nodes = try db.prepare(SelectNodesParams, SelectNodesResult,
        \\ SELECT minX AS x, minY AS y FROM nodes
    );
    defer select_nodes.finalize();

    var tree = quadtree.Quadtree.init(std.heap.c_allocator, area, .{});
    defer tree.deinit();

    var node_count: usize = 0;

    { // Insert nodes into Quadtree
        try select_nodes.bind(.{});
        defer select_nodes.reset();
        while (try select_nodes.step()) |result| {
            try tree.insert(.{ result.x, result.y }, 1.0);
            node_count += 1;
        }
    }

    std.log.info("inserted {d} points into quadtree", .{node_count});

    var tiles = std.ArrayList(Tile).init(std.heap.c_allocator);
    try addTile(&tiles, &tree, area, 0, 0);
    std.log.info("TILES", .{});
    for (tiles.items) |tile| {
        std.log.info(
            "- level {d}, count {d}, area [ c = ({d}, {d}), s = {d} ]",
            .{ tile.level, tile.count, tile.area.c[0], tile.area.c[1], tile.area.s },
        );
    }

    std.log.info("got {d} tiles", .{tiles.items.len});
}

const Tile = struct {
    level: u32,
    count: u32,
    area: quadtree.Area,
};

fn addTile(tiles: *std.ArrayList(Tile), tree: *const quadtree.Quadtree, area: quadtree.Area, id: u32, level: u32) !void {
    const node = tree.tree.items[id];
    const count: u32 = @intFromFloat(@round(node.mass));
    try tiles.append(.{ .level = level, .count = count, .area = area });

    if (count > config.capacity) {
        if (node.ne != quadtree.Node.NULL)
            try addTile(tiles, tree, area.divide(.ne), node.ne, level + 1);
        if (node.nw != quadtree.Node.NULL)
            try addTile(tiles, tree, area.divide(.nw), node.nw, level + 1);
        if (node.sw != quadtree.Node.NULL)
            try addTile(tiles, tree, area.divide(.sw), node.sw, level + 1);
        if (node.se != quadtree.Node.NULL)
            try addTile(tiles, tree, area.divide(.se), node.se, level + 1);
    }
}

fn getArea(db: sqlite.Database) !quadtree.Area {
    const SelectAreaParams = struct {};
    const SelectAreaResult = struct { min_x: f32, max_x: f32, min_y: f32, max_y: f32 };
    const select_area = try db.prepare(SelectAreaParams, SelectAreaResult,
        \\ SELECT
        \\   MIN(minX) AS min_x,
        \\   MAX(maxX) AS max_x,
        \\   MIN(minY) AS min_y,
        \\   MAX(maxY) AS max_y
        \\ FROM nodes
    );
    defer select_area.finalize();

    try select_area.bind(.{});
    defer select_area.reset();
    const result = try select_area.step() orelse return error.NoResults;

    const s = 2 * @max(@abs(result.min_x), @abs(result.max_x), @abs(result.min_y), @abs(result.max_y));
    return quadtree.Area{
        .c = .{ 0, 0 },
        // .s = s,
        .s = if (s > 0) std.math.pow(f32, 2, @ceil(std.math.log2(s))) else 0,
    };
}
