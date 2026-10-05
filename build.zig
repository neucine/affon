const std = @import("std");

fn addComputeBackends(compute_dep: *std.Build.Dependency, module: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag == .macos) {
        module.linkFramework("Accelerate", .{});
        module.linkFramework("Metal", .{});
        module.linkFramework("MetalPerformanceShaders", .{});
        module.linkFramework("Foundation", .{});
        module.addCSourceFile(.{ .file = compute_dep.path("src/backend/metal/ffi.m"), .flags = &.{"-fobjc-arc"} });
    } else module.addCSourceFile(.{ .file = compute_dep.path("src/backend/metal/ffi_stub.c"), .flags = &.{"-std=c11"} });
    if (target.result.os.tag == .linux) {
        module.addCSourceFile(.{ .file = compute_dep.path("src/backend/cuda/ffi.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
        module.linkSystemLibrary("dl", .{});
    } else module.addCSourceFile(.{ .file = compute_dep.path("src/backend/cuda/ffi_stub.c"), .flags = &.{"-std=c11"} });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const hao = b.dependency("hao", .{ .target = target, .optimize = optimize });
    const compute_dep = b.dependency("compute", .{ .target = target, .optimize = optimize });
    const zig_libs_dep = b.dependency("zig_libs", .{ .target = target, .optimize = optimize });
    const zig_libs = zig_libs_dep.module("zig_libs");
    const hao_module = hao.module("hao");
    hao_module.addImport("zig_libs", zig_libs);
    const compute = compute_dep.module("compute");

    _ = b.addModule("affon", .{
        .root_source_file = b.path("src/affon.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hao", .module = hao_module },
            .{ .name = "zig_libs", .module = zig_libs },
            .{ .name = "compute", .module = compute },
        },
    });

    const affon_module = b.modules.get("affon") orelse unreachable;
    const cli = b.addExecutable(.{
        .name = "affon",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "affon", .module = affon_module },
                .{ .name = "hao", .module = hao_module },
                .{ .name = "zig_libs", .module = zig_libs },
            },
        }),
    });
    cli.root_module.linkLibrary(hao.artifact("hao_runtime"));
    if (target.result.os.tag == .linux) {
        // ELF linking does not resolve the Rust archive nested in hao_runtime.
        cli.root_module.addObjectFile(hao.path("libs/transpiler/target/release/libhao_transpiler.a"));
        cli.root_module.linkSystemLibrary("unwind", .{});
    }
    addComputeBackends(compute_dep, cli.root_module, target);
    b.installArtifact(cli);

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/affon.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "hao", .module = hao_module },
                .{ .name = "zig_libs", .module = zig_libs },
                .{ .name = "compute", .module = compute },
            },
        }),
    });
    tests.root_module.linkLibrary(hao.artifact("hao_runtime"));
    if (target.result.os.tag == .linux) {
        tests.root_module.addObjectFile(hao.path("libs/transpiler/target/release/libhao_transpiler.a"));
        tests.root_module.linkSystemLibrary("unwind", .{});
    }
    addComputeBackends(compute_dep, tests.root_module, target);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Affon tests");
    test_step.dependOn(&run_tests.step);

}
