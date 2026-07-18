const std = @import("std");
const hao = @import("hao");

pub const package_name = "affon";
pub const compute = @import("compute/core.zig");
const autograd = @import("compute/autograd/index.zig");

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

test "compute graph execution fuses matmul and full-shape bias" {
    const Value = compute.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer rhs.deinit();
    const bias = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer bias.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, rhs);
    const bias_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, bias);
    var output_shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer output_shape.deinit();
    var output_layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, output_shape);
    defer output_layout.deinit();
    const output_spec = compute.types.tensor.ValueSpec{
        .shape = output_shape,
        .dtype = .f32,
        .layout = output_layout,
        .device = .cpu,
    };
    const matmul_id = try graph.addOp(.matmul, &.{ lhs_id, rhs_id }, .{ .none = {} }, output_spec);
    const output_id = try graph.addOp(.add, &.{ matmul_id, bias_id }, .{ .none = {} }, output_spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.execution.graph.execute(std.testing.allocator, &graph, &.{ lhs, rhs, bias });
    defer result.deinit();
    const bytes = try result.outputs[0].storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 23, 29, 50, 65 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute graph execution supports matmul add gelu epilogues" {
    const Value = compute.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer rhs.deinit();
    const bias = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer bias.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, rhs);
    const bias_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, bias);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.ValueSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const matmul_id = try graph.addOp(.matmul, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const add_id = try graph.addOp(.add, &.{ matmul_id, bias_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.gelu, &.{add_id}, .{ .none = {} }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.execution.graph.execute(std.testing.allocator, &graph, &.{ lhs, rhs, bias });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectEqual(@as(usize, 4), values.len);
    for (values) |value| try std.testing.expect(value > 0);
}

test "compute graph execution supports add layer norm fusion" {
    const Value = compute.tensor.Value;
    const lhs = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 3, 5, 7 });
    defer lhs.deinit();
    const rhs = try Value.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer rhs.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, rhs);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.ValueSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const sum_id = try graph.addOp(.add, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.layer_norm, &.{sum_id}, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.execution.graph.execute(std.testing.allocator, &graph, &.{ lhs, rhs });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0), values[0] + values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), values[2] + values[3], 1e-5);
}

test "compute graph execution supports attention score fusion" {
    const Value = compute.tensor.Value;
    const q = try Value.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 0, 0, 1 });
    defer q.deinit();
    const k_t = try Value.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 0, 0, 1 });
    defer k_t.deinit();
    const scale = try Value.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer scale.deinit();
    const mask = try Value.fromSliceI64(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 0, 0, 0, 0 });
    defer mask.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const q_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, q);
    const k_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, k_t);
    const scale_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, scale);
    const mask_id = try compute.execution.graph.builder.addInputFromValue(std.testing.allocator, &graph, mask);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 1, 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.ValueSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const mm_id = try graph.addOp(.matmul, &.{ q_id, k_id }, .{ .none = {} }, spec);
    const scaled_id = try graph.addOp(.mul, &.{ mm_id, scale_id }, .{ .none = {} }, spec);
    const masked_id = try graph.addOp(.masked_fill, &.{ scaled_id, mask_id }, .{ .masked_fill = .{ .value = -1e9 } }, spec);
    const output_id = try graph.addOp(.softmax, &.{masked_id}, .{ .softmax = .{ .axis = 2 } }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.execution.graph.execute(std.testing.allocator, &graph, &.{ q, k_t, scale, mask });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[0] + values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[2] + values[3], 1e-5);
}

test "autograd tape tracks provenance without coupling to execution" {
    const Value = compute.tensor.Value;
    const input = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    const output = try Value.fromSliceF32(std.testing.allocator, &.{2}, &.{ 3, 4 });

    _ = try autograd.makeTrainableValue(std.testing.allocator, input);
    _ = try autograd.makeTrackedValue(std.testing.allocator, output);
    const node = try autograd.tape.createNode(
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
    autograd.State.fromValue(output).?.attachNode(node);

    try std.testing.expectEqual(autograd.TrackingState.trainable, autograd.State.trackingState(input));
    try std.testing.expectEqual(autograd.TrackingState.tracked, autograd.State.trackingState(output));
    autograd.releaseOwnedValue(output);
    autograd.releaseOwnedValue(input);
}
