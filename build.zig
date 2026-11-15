const std = @import("std");

fn addWindowsUnwind(
    module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    unwind: ?[]const u8,
) void {
    if (target.result.os.tag != .windows) return;
    const unwind_path = unwind orelse return;
    const lib_dir = std.fs.path.dirname(unwind_path) orelse return;
    module.addLibraryPath(.{ .cwd_relative = lib_dir });
    module.linkSystemLibrary("mingwex", .{});
    module.addObjectFile(.{ .cwd_relative = unwind_path });
    module.addCSourceFile(.{ .file = .{ .cwd_relative = "src/windows_cfg_stub.c" }, .flags = &.{"-std=c11"} });
}

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

fn addComputeCynaLibraries(
    module: *std.Build.Module,
    compute_core_lib_dir: []const u8,
    cyna_lib_dir: []const u8,
) void {
    module.addLibraryPath(.{ .cwd_relative = compute_core_lib_dir });
    module.addLibraryPath(.{ .cwd_relative = cyna_lib_dir });
    module.linkSystemLibrary("compute-core", .{});
    module.linkSystemLibrary("cyna_std", .{});
    module.linkSystemLibrary("cyna", .{});
    module.linkSystemLibrary("m", .{});
    module.linkSystemLibrary("c", .{});
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const compute_cyna = b.option(
        bool,
        "compute-cyna",
        "Route covered CPU Program execution through compute-core and Cyna",
    ) orelse false;
    const compute_core_lib_dir = b.option(
        []const u8,
        "compute-core-lib-dir",
        "Directory containing compute-core's static library",
    ) orelse b.graph.environ_map.get("AFFON_COMPUTE_CORE_LIB_DIR") orelse "../compute-core/build";
    const cyna_lib_dir = b.option(
        []const u8,
        "cyna-lib-dir",
        "Directory containing the Cyna runtime static libraries",
    ) orelse b.graph.environ_map.get("AFFON_CYNA_LIB_DIR");
    if (compute_cyna and cyna_lib_dir == null) {
        @panic("-Dcompute-cyna requires -Dcyna-lib-dir, AFFON_CYNA_LIB_DIR, or explicit Cyna runtime libraries");
    }
    const windows_unwind = b.option(
        []const u8,
        "windows-gnu-libunwind",
        "Path to libunwind.a for x86_64-w64-mingw32",
    ) orelse b.graph.environ_map.get("AFFON_WINDOWS_GNU_LIBUNWIND");
    const transpiler_archive = if (target.result.os.tag == .windows)
        "libs/transpiler/target/x86_64-pc-windows-gnullvm/release/libhao_transpiler.a"
    else
        "libs/transpiler/target/release/libhao_transpiler.a";
    const hao = b.dependency("hao", .{ .target = target, .optimize = optimize });
    const compute_dep = b.dependency("compute", .{ .target = target, .optimize = optimize });
    const zig_libs_dep = b.dependency("zig_libs", .{ .target = target, .optimize = optimize });
    const zig_libs = zig_libs_dep.module("zig_libs");
    const hao_module = hao.module("hao");
    hao_module.addImport("zig_libs", zig_libs);
    const compute = compute_dep.module("compute");
    const options = b.addOptions();
    options.addOption(bool, "compute_cyna", compute_cyna);
    const build_options = options.createModule();

    _ = b.addModule("affon", .{
        .root_source_file = b.path("src/affon.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "hao", .module = hao_module },
            .{ .name = "zig_libs", .module = zig_libs },
            .{ .name = "compute", .module = compute },
            .{ .name = "build_options", .module = build_options },
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
    // Link the Rust archive at the final executable; a static library cannot
    // contain another archive as an object member.
    cli.root_module.addObjectFile(hao.path(transpiler_archive));
    if (target.result.os.tag == .linux) {
        cli.root_module.linkSystemLibrary("unwind", .{});
    }
    addWindowsUnwind(cli.root_module, target, windows_unwind);
    addComputeBackends(compute_dep, cli.root_module, target);
    if (compute_cyna) addComputeCynaLibraries(
        cli.root_module,
        compute_core_lib_dir,
        cyna_lib_dir.?,
    );
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
                .{ .name = "build_options", .module = build_options },
            },
        }),
    });
    tests.root_module.linkLibrary(hao.artifact("hao_runtime"));
    tests.root_module.addObjectFile(hao.path(transpiler_archive));
    if (target.result.os.tag == .linux) {
        tests.root_module.linkSystemLibrary("unwind", .{});
    }
    addWindowsUnwind(tests.root_module, target, windows_unwind);
    addComputeBackends(compute_dep, tests.root_module, target);
    if (compute_cyna) addComputeCynaLibraries(
        tests.root_module,
        compute_core_lib_dir,
        cyna_lib_dir.?,
    );
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Affon tests");
    test_step.dependOn(&run_tests.step);
}
