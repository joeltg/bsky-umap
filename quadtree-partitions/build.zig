const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // const sqlite_dep = b.dependency("sqlite", .{ .SQLITE_ENABLE_RTREE = true });
    // const sqlite = sqlite_dep.module("sqlite");

    const rtree_dep = b.dependency("rtree", .{});
    const quadtree = rtree_dep.module("quadtree");

    const cli_dep = b.dependency("cli", .{});

    const cli = b.addExecutable(.{
        .name = "quadtree-partitions",
        .root_source_file = b.path("./src/main.zig"),
        .optimize = optimize,
        .target = target,
    });

    // cli.root_module.addImport("sqlite", sqlite);
    cli.root_module.addImport("quadtree", quadtree);
    cli.root_module.addImport("zig-cli", cli_dep.module("zig-cli"));

    cli.linkLibC();
    b.installArtifact(cli);

    const run_artifact = b.addRunArtifact(cli);
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_artifact.step);
}
