const std = @import("std");

fn addMetalBackend(b: *std.Build, compute_dep: *std.Build.Dependency, module: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag == .macos) {
        const metal_compile = b.addSystemCommand(&.{ "xcrun", "metal", "-c" });
        metal_compile.addFileArg(compute_dep.path("src/pipeline/backend/metal/kernels.metal"));
        metal_compile.addArg("-o");
        const air_path = metal_compile.addOutputFileArg("affon_kernels.air");

        const metal_link = b.addSystemCommand(&.{ "xcrun", "metallib" });
        metal_link.addFileArg(air_path);
        metal_link.addArg("-o");
        const metallib_path = metal_link.addOutputFileArg("affon_kernels.metallib");

        const embed_metallib = b.addSystemCommand(&.{ "python3", b.path("tools/embed_binary.py").getPath(b) });
        embed_metallib.addFileArg(metallib_path);
        const metal_header = embed_metallib.addOutputFileArg("affon_metal_metallib.h");
        embed_metallib.addArg("affon_metal_metallib_data");

        module.linkFramework("Metal", .{});
        module.linkFramework("MetalPerformanceShaders", .{});
        module.linkFramework("Foundation", .{});
        module.addIncludePath(metal_header.dirname());
        module.addCSourceFile(.{
            .file = compute_dep.path("src/pipeline/backend/metal/ffi.m"),
            .flags = &.{"-fobjc-arc"},
        });
        b.getInstallStep().dependOn(&b.addInstallFile(metallib_path, "lib/affon_kernels.metallib").step);
    } else {
        module.addCSourceFile(.{
            .file = compute_dep.path("src/pipeline/backend/metal/ffi_stub.c"),
            .flags = &.{"-std=c11"},
        });
    }
}

fn addCudaBackend(compute_dep: *std.Build.Dependency, module: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag == .linux) {
        module.addCSourceFile(.{
            .file = compute_dep.path("src/pipeline/backend/cuda/ffi.c"),
            .flags = &.{ "-std=c11" },
        });
        module.linkSystemLibrary("dl", .{});
    } else {
        module.addCSourceFile(.{
            .file = compute_dep.path("src/pipeline/backend/cuda/ffi_stub.c"),
            .flags = &.{ "-std=c11" },
        });
    }
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
    if (target.result.os.tag == .linux) cli.root_module.linkSystemLibrary("unwind", .{});
    addMetalBackend(b, compute_dep, cli.root_module, target);
    addCudaBackend(compute_dep, cli.root_module, target);
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
    if (target.result.os.tag == .linux) tests.root_module.linkSystemLibrary("unwind", .{});
    addMetalBackend(b, compute_dep, tests.root_module, target);
    addCudaBackend(compute_dep, tests.root_module, target);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Affon tests");
    test_step.dependOn(&run_tests.step);
}
