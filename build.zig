const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module
    const lib_mod = b.addModule("exocortex_memory", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Static library
    const lib = b.addStaticLibrary(.{
        .name = "exocortex_memory",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Run step
    const exe = b.addExecutable(.{
        .name = "exocortex-memory-demo",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("exocortex_memory", lib_mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the demo");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Integration tests
    const test_files = &[_][]const u8{
        "tests/embedding_test.zig",
        "tests/insight_test.zig",
        "tests/store_test.zig",
        "tests/schema_test.zig",
        "tests/index_test.zig",
    };

    const test_step = b.step("test", "Run all tests");

    const run_unit = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit.step);

    for (test_files) |path| {
        const t = b.addTest(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = optimize,
        });
        t.root_module.addImport("exocortex_memory", lib_mod);
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }
}
