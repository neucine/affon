const std = @import("std");
const compute = @import("../../../compute/core.zig");
const diagnostic = @import("zig_libs").diagnostic;
const schema = @import("schema.zig");
const slice_semantics = @import("../../../compute/shared/sema/slice.zig");
const compose_builder = @import("../../../compute/compose/builder.zig");
const captured_op_info = @import("op_info.zig");
const ExecutionMetadata = @import("../../../compute/shared/types/operation/execution_metadata.zig").ExecutionMetadata;
const HintSource = @import("../../../compute/plan/matmul.zig").HintSource;
const GraphValueId = @import("../../../compute/shared/types/ir/index.zig").TensorId;

const TensorHandle = *compute.tensor.Tensor;
const Graph = @import("../../../compute/shared/types/ir/index.zig").Graph;
const CapturedMatmulExecutionJson = schema.CapturedMatmulExecutionJson;
const CapturedNodeJson = schema.CapturedNodeJson;
const CapturedProgramJson = schema.CapturedProgramJson;

fn parseDType(name: []const u8) !compute.tensor.DType {
    if (std.mem.eql(u8, name, "f32")) return .f32;
    if (std.mem.eql(u8, name, "f64")) return .f64;
    if (std.mem.eql(u8, name, "i64")) return .i64;
    return error.InvalidDType;
}

// Owns native lowerability analysis and lowering from serialized captured
// programs into `Graph`.
pub const CapturedGraphLoweringFailure = struct {
    category: []const u8,
    node_id: ?u32 = null,
    node_kind: ?[]const u8 = null,
    reason: []const u8,
};

pub const CapturedGraphLoweringAnalysis = struct {
    lowerable: bool,
    failure: ?CapturedGraphLoweringFailure = null,
};

pub const LoweredCapturedGraph = struct {
    graph: Graph,
    input_values: std.ArrayList(*compute.tensor.Tensor),
    owned_constant_values: std.ArrayList(?*compute.tensor.Tensor),
    owned_random_values: std.ArrayList(?*compute.tensor.Tensor),
    node_value_ids: std.AutoHashMap(u32, GraphValueId),

    pub fn deinit(self: *LoweredCapturedGraph, allocator: std.mem.Allocator) void {
        self.node_value_ids.deinit();
        for (self.owned_random_values.items) |value| if (value) |tensor| tensor.deinit();
        self.owned_random_values.deinit(allocator);
        for (self.owned_constant_values.items) |value| if (value) |tensor| tensor.deinit();
        self.owned_constant_values.deinit(allocator);
        self.input_values.deinit(allocator);
        self.graph.deinit();
        self.* = undefined;
    }

    pub fn takeOwnedValue(self: *LoweredCapturedGraph, value: *compute.tensor.Tensor) bool {
        for (self.owned_random_values.items) |*slot| {
            if (slot.* == value) {
                slot.* = null;
                return true;
            }
        }
        for (self.owned_constant_values.items) |*slot| {
            if (slot.* == value) {
                slot.* = null;
                return true;
            }
        }
        return false;
    }
};

pub const OpInfo = captured_op_info.OpInfo;

const lowering_failure_unsupported_node_kind = "unsupported_captured_node_kind";
const lowering_failure_invalid_adapter_metadata = "invalid_adapter_metadata";
const lowering_failure_invalid_execution_metadata = "invalid_execution_metadata";

fn loweringFailureForNode(node: ?CapturedNodeJson, category: []const u8, reason: []const u8) CapturedGraphLoweringFailure {
    return .{
        .category = category,
        .node_id = if (node) |n| n.id else null,
        .node_kind = if (node) |n| n.kind else null,
        .reason = reason,
    };
}

fn capturedMatmulExecutionMetadata(execution: ?CapturedMatmulExecutionJson) !ExecutionMetadata {
    const hint_name = execution orelse return .{};
    const hint = hint_name.hint orelse return .{};
    const hint_source = try capturedMatmulHintSource(hint_name.source);
    if (std.mem.eql(u8, hint, "projection")) {
        return .{ .matmul_hint = .projection, .hint_source = hint_source };
    }
    if (std.mem.eql(u8, hint, "attention_scores")) {
        return .{ .matmul_hint = .attention_scores, .hint_source = hint_source };
    }
    if (std.mem.eql(u8, hint, "attention_values")) {
        return .{ .matmul_hint = .attention_values, .hint_source = hint_source };
    }
    return error.InvalidExecutionMetadata;
}

fn capturedMatmulHintSource(source: ?[]const u8) !HintSource {
    const name = source orelse return .api_execution_arg;
    if (std.mem.eql(u8, name, "higher_level_module")) return .higher_level_module;
    if (std.mem.eql(u8, name, "api_execution_arg")) return .api_execution_arg;
    return error.InvalidExecutionMetadata;
}

fn capturedNodeInputIds(
    allocator: std.mem.Allocator,
    node_ids: *const std.AutoHashMap(u32, GraphValueId),
    node: CapturedNodeJson,
) ![]GraphValueId {
    if (std.mem.eql(u8, node.kind, "rand") or std.mem.eql(u8, node.kind, "randn")) {
        return allocator.alloc(GraphValueId, 0);
    }
    if (std.mem.eql(u8, node.kind, "neg") or std.mem.eql(u8, node.kind, "relu") or std.mem.eql(u8, node.kind, "abs") or std.mem.eql(u8, node.kind, "exp") or std.mem.eql(u8, node.kind, "log") or std.mem.eql(u8, node.kind, "sqrt") or std.mem.eql(u8, node.kind, "sigmoid") or std.mem.eql(u8, node.kind, "silu") or std.mem.eql(u8, node.kind, "tanh") or std.mem.eql(u8, node.kind, "sign") or std.mem.eql(u8, node.kind, "gelu") or std.mem.eql(u8, node.kind, "reshape") or std.mem.eql(u8, node.kind, "slice") or std.mem.eql(u8, node.kind, "squeeze") or std.mem.eql(u8, node.kind, "unsqueeze") or std.mem.eql(u8, node.kind, "softmax") or std.mem.eql(u8, node.kind, "transpose") or std.mem.eql(u8, node.kind, "permute") or std.mem.eql(u8, node.kind, "contiguous") or std.mem.eql(u8, node.kind, "sum") or std.mem.eql(u8, node.kind, "mean") or std.mem.eql(u8, node.kind, "std") or std.mem.eql(u8, node.kind, "variance") or std.mem.eql(u8, node.kind, "min") or std.mem.eql(u8, node.kind, "max") or std.mem.eql(u8, node.kind, "argmin") or std.mem.eql(u8, node.kind, "argmax") or std.mem.eql(u8, node.kind, "cast") or std.mem.eql(u8, node.kind, "one_hot") or std.mem.eql(u8, node.kind, "clamp")) {
        const input = node.input orelse return error.InvalidGraphPlan;
        const input_id = node_ids.get(input) orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 1);
        ids[0] = input_id;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "add") or std.mem.eql(u8, node.kind, "sub") or std.mem.eql(u8, node.kind, "mul") or std.mem.eql(u8, node.kind, "div") or std.mem.eql(u8, node.kind, "gt") or std.mem.eql(u8, node.kind, "dot") or std.mem.eql(u8, node.kind, "matmul")) {
        const left = node.left orelse return error.InvalidGraphPlan;
        const right = node.right orelse return error.InvalidGraphPlan;
        const left_id = node_ids.get(left) orelse return error.InvalidGraphPlan;
        const right_id = node_ids.get(right) orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 2);
        ids[0] = left_id;
        ids[1] = right_id;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "cross_entropy_indexed")) {
        const logits = node.logits orelse return error.InvalidGraphPlan;
        const targets = node.targets orelse return error.InvalidGraphPlan;
        const logits_id = node_ids.get(logits) orelse return error.InvalidGraphPlan;
        const targets_id = node_ids.get(targets) orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 2);
        ids[0] = logits_id;
        ids[1] = targets_id;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "where")) {
        const cond = node.cond orelse return error.InvalidGraphPlan;
        const on_true = node.onTrue orelse return error.InvalidGraphPlan;
        const on_false = node.onFalse orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 3);
        ids[0] = node_ids.get(cond) orelse return error.InvalidGraphPlan;
        ids[1] = node_ids.get(on_true) orelse return error.InvalidGraphPlan;
        ids[2] = node_ids.get(on_false) orelse return error.InvalidGraphPlan;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "masked_fill")) {
        const input = node.input orelse return error.InvalidGraphPlan;
        const mask = node.mask orelse return error.InvalidGraphPlan;
        const input_id = node_ids.get(input) orelse return error.InvalidGraphPlan;
        const mask_id = node_ids.get(mask) orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 2);
        ids[0] = input_id;
        ids[1] = mask_id;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "index_select")) {
        const input = node.input orelse return error.InvalidGraphPlan;
        const index = node.index orelse return error.InvalidGraphPlan;
        const input_id = node_ids.get(input) orelse return error.InvalidGraphPlan;
        const index_id = node_ids.get(@intCast(index)) orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 2);
        ids[0] = input_id;
        ids[1] = index_id;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "gather")) {
        const input = node.input orelse return error.InvalidGraphPlan;
        const index = node.index orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 2);
        ids[0] = node_ids.get(input) orelse return error.InvalidGraphPlan;
        ids[1] = node_ids.get(@intCast(index)) orelse return error.InvalidGraphPlan;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "topk_values") or std.mem.eql(u8, node.kind, "topk_indices")) {
        const input = node.input orelse return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, 1);
        ids[0] = node_ids.get(input) orelse return error.InvalidGraphPlan;
        return ids;
    }
    if (std.mem.eql(u8, node.kind, "cat") or std.mem.eql(u8, node.kind, "stack")) {
        const input_ids = node.inputs orelse return error.InvalidGraphPlan;
        if (input_ids.len == 0) return error.InvalidGraphPlan;
        const ids = try allocator.alloc(GraphValueId, input_ids.len);
        for (input_ids, ids) |input_id, *id| id.* = node_ids.get(input_id) orelse return error.InvalidGraphPlan;
        return ids;
    }
    return error.UnsupportedGraphLowering;
}

pub fn nodeTagAndOptions(node: CapturedNodeJson) !OpInfo {
    if (captured_op_info.noOptionKind(node.kind)) |info| return info;
    if (std.mem.eql(u8, node.kind, "reshape")) return .{ .tag = .reshape, .options = .{ .reshape = .{ .shape = node.shape orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "slice")) return .{ .tag = .slice, .options = .{ .slice = .{ .ranges = node.slice_ranges orelse if (node.slice_selectors != null) &.{} else return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "squeeze")) return .{ .tag = .squeeze, .options = .{ .squeeze = .{ .axis = node.axis } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "unsqueeze")) return .{ .tag = .unsqueeze, .options = .{ .unsqueeze = .{ .axis = node.axis orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "softmax")) return captured_op_info.softmax(node.dim orelse node.axis orelse return error.InvalidGraphPlan);
    if (captured_op_info.reduction(node.kind, node.axis, node.keepdim orelse false)) |info| return info;
    if (std.mem.eql(u8, node.kind, "transpose")) return captured_op_info.transpose();
    if (std.mem.eql(u8, node.kind, "permute")) return .{ .tag = .permute, .options = .{ .permute = .{ .axes = node.axes orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "cast")) return .{ .tag = .cast, .options = .{ .cast = .{ .to = try parseDType(node.dtype orelse return error.InvalidGraphPlan) } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "one_hot")) return .{ .tag = .one_hot, .options = .{ .one_hot = .{ .num_classes = node.numClasses orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "clamp")) return .{ .tag = .clamp, .options = .{ .clamp = .{ .min = node.min orelse return error.InvalidGraphPlan, .max = node.max orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "gather")) return captured_op_info.gather(node.dim orelse return error.InvalidGraphPlan);
    if (std.mem.eql(u8, node.kind, "topk_values") or std.mem.eql(u8, node.kind, "topk_indices")) {
        return captured_op_info.topk(
            node.kind,
            node.k orelse return error.InvalidGraphPlan,
            node.dim orelse return error.InvalidGraphPlan,
            true,
            true,
        ) orelse error.UnsupportedGraphLowering;
    }
    if (std.mem.eql(u8, node.kind, "cat")) return captured_op_info.concat(node.dim orelse return error.InvalidGraphPlan);
    if (std.mem.eql(u8, node.kind, "stack")) return captured_op_info.stack(node.dim orelse return error.InvalidGraphPlan);
    if (std.mem.eql(u8, node.kind, "masked_fill")) return .{ .tag = .masked_fill, .options = .{ .masked_fill = .{ .value = node.value orelse return error.InvalidGraphPlan } }, .execution_metadata = .{} };
    if (std.mem.eql(u8, node.kind, "index_select")) return captured_op_info.indexSelect(node.dim orelse return error.InvalidGraphPlan);
    if (std.mem.eql(u8, node.kind, "matmul")) return .{ .tag = .matmul, .options = .{ .none = {} }, .execution_metadata = try capturedMatmulExecutionMetadata(node.execution) };
    if (std.mem.eql(u8, node.kind, "cross_entropy_indexed")) return captured_op_info.crossEntropyIndexed(node.axis orelse return error.InvalidGraphPlan);
    return error.UnsupportedGraphLowering;
}

fn readSliceSelectors(allocator: std.mem.Allocator, node: CapturedNodeJson) ![]slice_semantics.SliceSelector {
    const source = node.slice_selectors orelse return error.InvalidGraphPlan;
    const selectors = try allocator.alloc(slice_semantics.SliceSelector, source.len);
    errdefer allocator.free(selectors);
    for (source, selectors) |selector, *out| {
        if (std.mem.eql(u8, selector.kind, "all")) {
            out.* = .all;
        } else if (std.mem.eql(u8, selector.kind, "index")) {
            out.* = .{ .index = selector.index orelse return error.InvalidGraphPlan };
        } else if (std.mem.eql(u8, selector.kind, "range")) {
            out.* = .{ .range = .{
                .start = selector.start,
                .stop = selector.stop,
                .step = selector.step orelse 1,
            } };
        } else {
            return error.InvalidGraphPlan;
        }
    }
    return selectors;
}

fn sliceOptionsForNode(allocator: std.mem.Allocator, node: CapturedNodeJson, input_shape: []const usize) !compute.operation.OpOptions {
    if (node.slice_selectors != null) {
        const selectors = try readSliceSelectors(allocator, node);
        defer allocator.free(selectors);
        const ranges = try slice_semantics.normalizeSelectors(allocator, input_shape, selectors);
        return .{ .slice = .{ .ranges = ranges } };
    }
    return .{ .slice = .{ .ranges = try allocator.dupe(compute.operation.SliceRange, node.slice_ranges orelse return error.InvalidGraphPlan) } };
}

fn deinitLoweringOpOptions(allocator: std.mem.Allocator, options: compute.operation.OpOptions) void {
    switch (options) {
        .slice => |slice| allocator.free(slice.ranges),
        else => {},
    }
}

pub fn analyzeLowerability(
    program: CapturedProgramJson,
    comptime parseDTypeFn: fn ([]const u8) anyerror!compute.tensor.DType,
) CapturedGraphLoweringAnalysis {
    for (program.nodes) |node| {
        if (std.mem.eql(u8, node.kind, "input")) {
            if (node.index == null) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "input node missing index") };
            continue;
        }
        if (std.mem.eql(u8, node.kind, "constant")) {
            if (node.data == null) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "constant node is missing scalar data") };
            if (node.dtype == null) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "constant node is missing dtype") };
            _ = parseDTypeFn(node.dtype.?) catch {
                return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "constant node has unsupported dtype") };
            };
            continue;
        }
        if (std.mem.eql(u8, node.kind, "rand") or std.mem.eql(u8, node.kind, "randn")) {
            if (node.shape == null) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "random node is missing shape") };
            if (node.dtype == null) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "random node is missing dtype") };
            const dtype = parseDTypeFn(node.dtype.?) catch {
                return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "random node has unsupported dtype") };
            };
            if (dtype == .i64) return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "random nodes only support floating-point dtypes") };
            continue;
        }
        _ = nodeTagAndOptions(node) catch |err| switch (err) {
            error.UnsupportedGraphLowering => return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_unsupported_node_kind, "captured node kind is not yet lowerable") },
            error.InvalidGraphPlan => return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "captured node is missing lowering metadata") },
            error.InvalidExecutionMetadata => return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_execution_metadata, "captured node has unsupported execution metadata") },
            error.InvalidDType => return .{ .lowerable = false, .failure = loweringFailureForNode(node, lowering_failure_invalid_adapter_metadata, "captured node has unsupported dtype") },
        };
    }
    return .{ .lowerable = true };
}

pub fn lowerToGraph(
    allocator: std.mem.Allocator,
    program: CapturedProgramJson,
    input_handles: []const TensorHandle,
    comptime parseDTypeFn: fn ([]const u8) anyerror!compute.tensor.DType,
    comptime createScalarFn: fn (compute.tensor.DType, f64) anyerror!*compute.tensor.Tensor,
    comptime createRandomFn: fn ([]const usize, compute.tensor.DType, bool) anyerror!*compute.tensor.Tensor,
) !LoweredCapturedGraph {
    if (input_handles.len != program.inputCount) return error.InputCountMismatch;

    var lowered: LoweredCapturedGraph = .{
        .graph = Graph.init(allocator),
        .input_values = .empty,
        .owned_constant_values = .empty,
        .owned_random_values = .empty,
        .node_value_ids = std.AutoHashMap(u32, GraphValueId).init(allocator),
    };
    errdefer lowered.deinit(allocator);

    for (program.nodes) |node| {
        if (std.mem.eql(u8, node.kind, "input")) {
            const index = node.index orelse return error.InvalidGraphPlan;
            if (index >= input_handles.len) return error.IndexOutOfBounds;
            const input_value = input_handles[index];
            const spec = try input_value.spec();
            const graph_value_id = try lowered.graph.addInput(spec);
            try lowered.input_values.append(allocator, input_value);
            try lowered.node_value_ids.put(node.id, graph_value_id);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "constant")) {
            const scalar = node.data orelse return error.InvalidGraphPlan;
            const dtype_name = node.dtype orelse return error.InvalidGraphPlan;
            const dtype = try parseDTypeFn(dtype_name);
            const constant_value = try createScalarFn(dtype, scalar);
            try lowered.owned_constant_values.append(allocator, constant_value);
            const spec = try constant_value.spec();
            const graph_value_id = try lowered.graph.addInput(spec);
            try lowered.input_values.append(allocator, constant_value);
            try lowered.node_value_ids.put(node.id, graph_value_id);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "rand") or std.mem.eql(u8, node.kind, "randn")) {
            const shape = node.shape orelse return error.InvalidGraphPlan;
            const dtype_name = node.dtype orelse return error.InvalidGraphPlan;
            const dtype = try parseDTypeFn(dtype_name);
            const random_value = try createRandomFn(shape, dtype, std.mem.eql(u8, node.kind, "randn"));
            try lowered.owned_random_values.append(allocator, random_value);
            const spec = try random_value.spec();
            const graph_value_id = try lowered.graph.addInput(spec);
            try lowered.input_values.append(allocator, random_value);
            try lowered.node_value_ids.put(node.id, graph_value_id);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "topk_indices")) {
            if (lowered.node_value_ids.contains(node.id)) continue;
            return error.InvalidGraphPlan;
        }

        var node_inputs = try capturedNodeInputIds(allocator, &lowered.node_value_ids, node);
        defer allocator.free(node_inputs);

        if (std.mem.eql(u8, node.kind, "topk_values")) {
            const op_info = try nodeTagAndOptions(node);
            const input_specs = try allocator.alloc(compute.tensor.TensorSpec, node_inputs.len);
            defer allocator.free(input_specs);
            for (node_inputs, 0..) |input_id, i| input_specs[i] = lowered.graph.values.items[input_id].spec;
            const output_ids = try compose_builder.addOpFromSpecs(allocator, &lowered.graph, op_info.tag, node_inputs, input_specs, op_info.options, op_info.execution_metadata);
            defer allocator.free(output_ids);
            if (output_ids.len != 2) return error.InvalidGraphPlan;
            try lowered.node_value_ids.put(node.id, output_ids[0]);
            try lowered.node_value_ids.put(node.id + 1, output_ids[1]);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "masked_fill") and node_inputs.len == 2) {
            const mask_input_id = node_inputs[1];
            const mask_spec = lowered.graph.values.items[mask_input_id].spec;
            if (mask_spec.dtype != .i64) {
                var cast_shape = try compute.tensor.Shape.initCopy(allocator, mask_spec.shape.dims);
                errdefer cast_shape.deinit();
                var cast_layout = try compute.tensor.Layout.initCopy(allocator, mask_spec.layout.strides, mask_spec.layout.offset);
                errdefer cast_layout.deinit();
                const casted_mask = try lowered.graph.addOp(.cast, &.{mask_input_id}, .{ .cast = .{ .to = .i64 } }, .{
                    .shape = cast_shape,
                    .dtype = .i64,
                    .layout = cast_layout,
                    .device = mask_spec.device,
                    .axes = mask_spec.axes,
                });
                const rewritten_inputs = try allocator.dupe(GraphValueId, node_inputs);
                allocator.free(node_inputs);
                node_inputs = rewritten_inputs;
                node_inputs[1] = casted_mask;
            }
        }
        if (std.mem.eql(u8, node.kind, "index_select") and node_inputs.len == 2) {
            const index_input_id = node_inputs[1];
            const index_spec = lowered.graph.values.items[index_input_id].spec;
            if (index_spec.dtype != .i64) {
                var cast_shape = try compute.tensor.Shape.initCopy(allocator, index_spec.shape.dims);
                errdefer cast_shape.deinit();
                var cast_layout = try compute.tensor.Layout.initCopy(allocator, index_spec.layout.strides, index_spec.layout.offset);
                errdefer cast_layout.deinit();
                const casted_index = try lowered.graph.addOp(.cast, &.{index_input_id}, .{ .cast = .{ .to = .i64 } }, .{
                    .shape = cast_shape,
                    .dtype = .i64,
                    .layout = cast_layout,
                    .device = index_spec.device,
                    .axes = index_spec.axes,
                });
                const rewritten_inputs = try allocator.dupe(GraphValueId, node_inputs);
                allocator.free(node_inputs);
                node_inputs = rewritten_inputs;
                node_inputs[1] = casted_index;
            }
        }
        if (std.mem.eql(u8, node.kind, "cross_entropy_indexed") and node_inputs.len == 2) {
            const target_input_id = node_inputs[1];
            const target_spec = lowered.graph.values.items[target_input_id].spec;
            if (target_spec.dtype != .i64) {
                var cast_shape = try compute.tensor.Shape.initCopy(allocator, target_spec.shape.dims);
                errdefer cast_shape.deinit();
                var cast_layout = try compute.tensor.Layout.initCopy(allocator, target_spec.layout.strides, target_spec.layout.offset);
                errdefer cast_layout.deinit();
                const casted_target = try lowered.graph.addOp(.cast, &.{target_input_id}, .{ .cast = .{ .to = .i64 } }, .{
                    .shape = cast_shape,
                    .dtype = .i64,
                    .layout = cast_layout,
                    .device = target_spec.device,
                    .axes = target_spec.axes,
                });
                const rewritten_inputs = try allocator.dupe(GraphValueId, node_inputs);
                allocator.free(node_inputs);
                node_inputs = rewritten_inputs;
                node_inputs[1] = casted_target;
            }
        }

        const input_specs = try allocator.alloc(compute.tensor.TensorSpec, node_inputs.len);
        defer allocator.free(input_specs);
        for (node_inputs, 0..) |input_id, i| input_specs[i] = lowered.graph.values.items[input_id].spec;
        var owns_op_options = false;
        const op_info: OpInfo = if (std.mem.eql(u8, node.kind, "slice")) blk: {
            if (input_specs.len != 1) return error.InvalidGraphPlan;
            const options = try sliceOptionsForNode(allocator, node, input_specs[0].shape.dims);
            owns_op_options = true;
            break :blk .{ .tag = .slice, .options = options, .execution_metadata = .{} };
        } else try nodeTagAndOptions(node);
        defer if (owns_op_options) deinitLoweringOpOptions(allocator, op_info.options);

        const output_ids = compose_builder.addOpFromSpecs(allocator, &lowered.graph, op_info.tag, node_inputs, input_specs, op_info.options, op_info.execution_metadata) catch |err| {
            return diagnostic.withError(
                err,
                "captured.lowering: add op failed for node {d} kind={s} op={s}",
                .{ node.id, node.kind, @tagName(op_info.tag) },
            );
        };
        defer allocator.free(output_ids);
        if (output_ids.len != 1) return error.InvalidGraphPlan;
        const graph_value_id = output_ids[0];
        if (node.module_path) |module_path| {
            lowered.graph.nodes.items[lowered.graph.nodes.items.len - 1].module_path = try allocator.dupe(u8, module_path);
        }
        try lowered.node_value_ids.put(node.id, graph_value_id);
    }

    const output_value_id = lowered.node_value_ids.get(program.outputId) orelse return error.InvalidGraphPlan;
    try lowered.graph.setOutputs(&.{output_value_id});
    return lowered;
}

pub fn buildInputHandles(
    allocator: std.mem.Allocator,
    program: CapturedProgramJson,
    lowered: *const LoweredCapturedGraph,
    input_handles: []const TensorHandle,
) ![]TensorHandle {
    const handles = try allocator.alloc(TensorHandle, lowered.graph.inputs.items.len);
    errdefer allocator.free(handles);
    var next_input: usize = 0;
    var next_constant: usize = 0;
    var next_random: usize = 0;
    for (program.nodes) |node| {
        if (std.mem.eql(u8, node.kind, "input")) {
            const index = node.index orelse return error.InvalidGraphPlan;
            if (index >= input_handles.len) return error.IndexOutOfBounds;
            handles[next_input] = input_handles[index];
            next_input += 1;
            continue;
        }
        if (std.mem.eql(u8, node.kind, "constant")) {
            if (next_constant >= lowered.owned_constant_values.items.len) return error.InvalidGraphPlan;
            handles[next_input] = lowered.owned_constant_values.items[next_constant] orelse return error.InvalidGraphPlan;
            next_input += 1;
            next_constant += 1;
            continue;
        }
        if (std.mem.eql(u8, node.kind, "rand") or std.mem.eql(u8, node.kind, "randn")) {
            if (next_random >= lowered.owned_random_values.items.len) return error.InvalidGraphPlan;
            handles[next_input] = lowered.owned_random_values.items[next_random] orelse return error.InvalidGraphPlan;
            next_input += 1;
            next_random += 1;
        }
    }
    if (next_input != lowered.graph.inputs.items.len) return error.InvalidGraphPlan;
    return handles;
}
