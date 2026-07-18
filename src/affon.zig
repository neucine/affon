const std = @import("std");
const hao = @import("hao");

pub const package_name = "affon";
pub const compute = @import("compute/core.zig");

const sources = [_]hao.SourceModule{.{
    .specifier = "affon:runtime",
    .source =
    \\export const name = "affon";
    \\export const version = "0.1.0";
    ,
}};

pub fn register(environment: *hao.RuntimeEnvironment) !void {
    try environment.registerPackage(.{
        .name = package_name,
        .sources = &sources,
    });
}

test "registers the Affon package through Hao" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();

    try register(&environment);
    try environment.evalModuleSource(
        "import { name } from 'affon:runtime'; globalThis.__affon_name = name;",
        "<affon-test>",
    );

    const global = hao.qjs.c.JS_GetGlobalObject(environment.runtime.ctx);
    defer hao.qjs.freeValue(environment.runtime.ctx, global);
    const value = hao.qjs.getProperty(environment.runtime.ctx, global, "__affon_name");
    defer hao.qjs.freeValue(environment.runtime.ctx, value);
    const name = try hao.qjs.valueToStringAlloc(environment.runtime.ctx, value, std.testing.allocator);
    defer std.testing.allocator.free(name);
    try std.testing.expectEqualStrings("affon", name);
}

test "compute core owns tensor values independently of the JS binding" {
    const Value = compute.tensor.Value;
    var value = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, value.shape.dims);
    try std.testing.expectEqual(compute.tensor.DType.f32, value.dtype);
    const bytes = try value.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqual(@as(f32, 3), values[2]);
}
