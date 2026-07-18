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
    const Value = compute.types.tensor.Value;
    var value = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, value.shape.dims);
    try std.testing.expectEqual(compute.types.tensor.DType.f32, value.dtype);
    const bytes = try value.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqual(@as(f32, 3), values[2]);
}

test "compute CPU kernels consume tensor storage" {
    const Value = compute.types.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();
    const result = try Value.createContiguous(std.testing.allocator, &.{3}, .f32, .cpu, false);
    defer result.deinit();

    try compute.backend.cpu.add(.f32, lhs.storage.?, rhs.storage.?, result.storage.?);
    const bytes = try result.storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute eager execution runs an operation" {
    const Value = compute.types.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();

    const inputs = [_]*Value{ lhs, rhs };
    const op = try compute.operation.Op.init(.add, &inputs, .{ .none = {} });
    const result = try compute.execution.eager.execute(std.testing.allocator, op);
    defer result.deinit();

    const bytes = try result.storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute graph execution runs multiple operations" {
    const Value = compute.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ -1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{3}, &.{ 2, 3, 4 });
    defer rhs.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, rhs);
    const spec = try lhs.spec();
    const sum_id = try graph.addOp(.add, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.relu, &.{sum_id}, .{ .none = {} }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.execution.graph.execute(std.testing.allocator, &graph, &.{ lhs, rhs });
    defer result.deinit();
    const bytes = try result.outputs[0].storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 1, 5, 7 }, std.mem.bytesAsSlice(f32, bytes));
}

test "autograd tape tracks provenance without coupling to execution" {
    const Value = compute.tensor.Value;
    const input = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    const output = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 3, 4 });

    _ = try compute.autograd.makeTrainableValue(std.testing.allocator, input);
    _ = try compute.autograd.makeTrackedValue(std.testing.allocator, output);
    const node = try compute.autograd.tape.createNode(
        std.testing.allocator,
        .relu,
        &.{.{ .value = input, .input_slot = 0 }},
        &.{input},
        output,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
    );
    compute.autograd.State.fromValue(output).?.attachNode(node);

    try std.testing.expectEqual(compute.autograd.TrackingState.trainable, compute.autograd.State.trackingState(input));
    try std.testing.expectEqual(compute.autograd.TrackingState.tracked, compute.autograd.State.trackingState(output));
    compute.autograd.releaseOwnedValue(output);
    compute.autograd.releaseOwnedValue(input);
}
