const std = @import("std");
const compute = @import("compute");
const OpTag = compute.operation.OpTag;
const OpOptions = compute.operation.OpOptions;
const ExecutionMetadata = compute.operation.ExecutionMetadata;

/// Shared captured-op vocabulary for JS option inference and captured JSON
/// lowering. Callers still own reading their source representation; this module
/// only maps validated captured names/options onto compute operation tags.
pub const OpInfo = struct {
    tag: OpTag,
    options: OpOptions,
    execution_metadata: ExecutionMetadata = .{},
};

fn noOption(tag: OpTag) OpInfo {
    return .{ .tag = tag, .options = .{ .none = {} } };
}

pub fn noOptionKind(kind: []const u8) ?OpInfo {
    if (std.mem.eql(u8, kind, "neg")) return noOption(.neg);
    if (std.mem.eql(u8, kind, "relu")) return noOption(.relu);
    if (std.mem.eql(u8, kind, "abs")) return noOption(.abs);
    if (std.mem.eql(u8, kind, "exp")) return noOption(.exp);
    if (std.mem.eql(u8, kind, "log")) return noOption(.log);
    if (std.mem.eql(u8, kind, "sqrt")) return noOption(.sqrt);
    if (std.mem.eql(u8, kind, "sigmoid")) return noOption(.sigmoid);
    if (std.mem.eql(u8, kind, "silu")) return noOption(.silu);
    if (std.mem.eql(u8, kind, "erf")) return noOption(.erf);
    if (std.mem.eql(u8, kind, "tanh")) return noOption(.tanh);
    if (std.mem.eql(u8, kind, "sign")) return noOption(.sign);
    if (std.mem.eql(u8, kind, "gelu")) return noOption(.gelu);
    if (std.mem.eql(u8, kind, "add")) return noOption(.add);
    if (std.mem.eql(u8, kind, "sub")) return noOption(.sub);
    if (std.mem.eql(u8, kind, "mul")) return noOption(.mul);
    if (std.mem.eql(u8, kind, "div")) return noOption(.div);
    if (std.mem.eql(u8, kind, "gt")) return noOption(.gt);
    if (std.mem.eql(u8, kind, "dot")) return noOption(.dot);
    if (std.mem.eql(u8, kind, "where")) return noOption(.where);
    if (std.mem.eql(u8, kind, "contiguous")) return noOption(.contiguous);
    return null;
}

pub fn reduction(kind: []const u8, axis: ?usize, keepdim: bool) ?OpInfo {
    const tags = reductionTags(kind) orelse return null;
    if (axis) |value| {
        return .{ .tag = tags.axis, .options = .{ .reduce_axis = .{ .axis = value, .keepdim = keepdim } } };
    }
    return .{ .tag = tags.all, .options = .{ .reduce_all = .{ .keepdim = keepdim } } };
}

const ReductionTags = struct { all: OpTag, axis: OpTag };

fn reductionTags(kind: []const u8) ?ReductionTags {
    if (std.mem.eql(u8, kind, "mean")) return .{ .all = .mean_all, .axis = .mean_axis };
    if (std.mem.eql(u8, kind, "variance")) return .{ .all = .variance_all, .axis = .variance_axis };
    if (std.mem.eql(u8, kind, "sum")) return .{ .all = .sum_all, .axis = .sum_axis };
    if (std.mem.eql(u8, kind, "min")) return .{ .all = .min_all, .axis = .min_axis };
    if (std.mem.eql(u8, kind, "max")) return .{ .all = .max_all, .axis = .max_axis };
    if (std.mem.eql(u8, kind, "std")) return .{ .all = .std_all, .axis = .std_axis };
    if (std.mem.eql(u8, kind, "argmin")) return .{ .all = .argmin_all, .axis = .argmin_axis };
    if (std.mem.eql(u8, kind, "argmax")) return .{ .all = .argmax_all, .axis = .argmax_axis };
    return null;
}

pub fn transpose() OpInfo {
    return .{ .tag = .transpose, .options = .{ .transpose = .{} } };
}

pub fn softmax(axis: usize) OpInfo {
    return .{ .tag = .softmax, .options = .{ .softmax = .{ .axis = axis } } };
}

pub fn crossEntropyIndexed(axis: usize) OpInfo {
    return .{ .tag = .cross_entropy_indexed, .options = .{ .cross_entropy_indexed = .{ .axis = axis } } };
}

pub fn gather(axis: usize) OpInfo {
    return .{ .tag = .gather, .options = .{ .gather = .{ .axis = axis } } };
}

pub fn indexSelect(axis: usize) OpInfo {
    return .{ .tag = .index_select, .options = .{ .index_select = .{ .axis = axis } } };
}

pub fn concat(axis: usize) OpInfo {
    return .{ .tag = .cat, .options = .{ .concat = .{ .axis = axis } } };
}

pub fn stack(axis: usize) OpInfo {
    return .{ .tag = .stack, .options = .{ .stack = .{ .axis = axis } } };
}

pub fn topk(kind: []const u8, k: usize, axis: usize, largest: bool, sorted: bool) ?OpInfo {
    if (!std.mem.eql(u8, kind, "topk_values") and !std.mem.eql(u8, kind, "topk_indices")) return null;
    return .{ .tag = .topk, .options = .{ .topk = .{ .k = k, .axis = axis, .largest = largest, .sorted = sorted } } };
}
