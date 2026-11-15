const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const node_include = b.option([]const u8, "node-include", "Directory containing node_api.h") orelse @panic("pass -Dnode-include=/path/to/node/include/node");
    const dep = b.dependency("compute", .{ .target = target, .optimize = optimize });
    const addon = b.addLibrary(.{
        .name = "ownership",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("bridge.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "compute", .module = dep.module("compute") }},
        }),
    });
    addon.root_module.addCSourceFile(.{ .file = b.path("addon.c"), .flags = &.{"-std=c11"} });
    addon.root_module.addIncludePath(.{ .cwd_relative = node_include });
    addon.root_module.link_libc = true;
    addon.linker_allow_shlib_undefined = true;
    if (target.result.os.tag == .macos) {
        addon.root_module.linkFramework("Accelerate", .{});
        addon.root_module.linkFramework("Metal", .{});
        addon.root_module.linkFramework("MetalPerformanceShaders", .{});
        addon.root_module.linkFramework("Foundation", .{});
        addon.root_module.addCSourceFile(.{ .file = dep.path("src/backend/metal/ffi.m"), .flags = &.{"-fobjc-arc"} });
    } else addon.root_module.addCSourceFile(.{ .file = dep.path("src/backend/metal/ffi_stub.c"), .flags = &.{"-std=c11"} });
    if (target.result.os.tag == .linux) {
        addon.root_module.addCSourceFile(.{ .file = dep.path("src/backend/cuda/ffi.c"), .flags = &.{"-std=c11"} });
        addon.root_module.linkSystemLibrary("dl", .{});
    } else addon.root_module.addCSourceFile(.{ .file = dep.path("src/backend/cuda/ffi_stub.c"), .flags = &.{"-std=c11"} });
    const install = b.addInstallArtifact(addon, .{ .dest_sub_path = "ownership.node" });
    b.getInstallStep().dependOn(&install.step);
}
