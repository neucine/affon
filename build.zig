const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const hao = b.dependency("hao", .{ .target = target, .optimize = optimize });

    _ = b.addModule("affon", .{
        .root_source_file = b.path("src/affon.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "hao", .module = hao.module("hao") }},
    });

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/affon.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "hao", .module = hao.module("hao") }},
        }),
    });
    tests.root_module.linkLibrary(hao.artifact("hao_runtime"));
    tests.root_module.addCSourceFile(.{
          .file = b.path("src/compute/backend/metal/ffi_stub.c"),
        .flags = &.{"-std=c11"},
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Affon tests");
    test_step.dependOn(&run_tests.step);
}
