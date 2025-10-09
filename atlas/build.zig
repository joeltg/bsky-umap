const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const sqlite_dep = b.dependency("sqlite", .{});
    const sqlite = sqlite_dep.module("sqlite");

    const rtree_dep = b.dependency("rtree", .{});
    const quadtree = rtree_dep.module("quadtree");

    const cli_dep = b.dependency("cli", .{});
    const cli = cli_dep.module("cli");

    const atlas = b.addModule("atlas", .{
        .root_source_file = b.path("./src/Atlas.zig"),
    });

    {
        const lib = b.addExecutable(.{
            .name = "wasmtest",
            .version = .{ .major = 0, .minor = 0, .patch = 1 },
            .root_module = b.createModule(.{
                .root_source_file = b.path("./lib/lib.zig"),
                .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
                .optimize = optimize,
            }),
        });

        lib.root_module.addImport("atlas", atlas);

        lib.entry = .disabled;
        lib.rdynamic = true;
        b.installArtifact(lib);
    }

    {
        const exe = b.addExecutable(.{
            .name = "quadtree-partitions",
            .root_module = b.createModule(.{
                .root_source_file = b.path("./src/main.zig"),
                .optimize = optimize,
                .target = target,
            }),
        });

        exe.root_module.addImport("sqlite", sqlite);
        exe.root_module.addImport("quadtree", quadtree);
        exe.root_module.addImport("zig-cli", cli);

        exe.linkLibC();
        b.installArtifact(exe);

        const run_artifact = b.addRunArtifact(exe);
        const run_step = b.step("run", "Run the application");
        run_step.dependOn(&run_artifact.step);
    }
}
