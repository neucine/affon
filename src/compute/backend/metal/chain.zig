const std = @import("std");
const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const OpTag = @import("../../types/operation/tag.zig").OpTag;
const common = @import("common.zig");

pub const UnaryStage = struct {
    tag: OpTag,
    clamp_min: ?f64 = null,
    clamp_max: ?f64 = null,
};

extern fn affon_metal_unary_chain_f32(input_handle: *anyopaque, out_handle: *anyopaque, op_codes: [*]const u32, clamp_mins: [*]const f32, clamp_maxs: [*]const f32, stage_count: usize, len: usize) c_int;
extern fn affon_metal_unary_chain_i64(input_handle: *anyopaque, out_handle: *anyopaque, op_codes: [*]const u32, clamp_mins: [*]const i64, clamp_maxs: [*]const i64, stage_count: usize, len: usize) c_int;
extern fn affon_metal_binary_then_unary_chain_f32(lhs_handle: *anyopaque, rhs_handle: *anyopaque, out_handle: *anyopaque, binary_code: u32, op_codes: [*]const u32, clamp_mins: [*]const f32, clamp_maxs: [*]const f32, stage_count: usize, len: usize) c_int;
extern fn affon_metal_binary_then_unary_chain_i64(lhs_handle: *anyopaque, rhs_handle: *anyopaque, out_handle: *anyopaque, binary_code: u32, op_codes: [*]const u32, clamp_mins: [*]const i64, clamp_maxs: [*]const i64, stage_count: usize, len: usize) c_int;

pub fn unaryChain(dtype: DType, input: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (stages.len == 0) return error.InvalidInputCount;
    if (stages.len > 16) return error.ExecutionNotImplemented;

    var op_codes: [16]u32 = [_]u32{0} ** 16;
    var mins_f32: [16]f32 = [_]f32{0} ** 16;
    var maxs_f32: [16]f32 = [_]f32{0} ** 16;
    var mins_i64: [16]i64 = [_]i64{0} ** 16;
    var maxs_i64: [16]i64 = [_]i64{0} ** 16;
    try encodeStages(stages, &op_codes, &mins_f32, &maxs_f32, &mins_i64, &maxs_i64);

    const rc = switch (dtype) {
        .f32 => affon_metal_unary_chain_f32(try input.metalHandle(), try out.metalHandle(), &op_codes, &mins_f32, &maxs_f32, stages.len, out.bytes / @sizeOf(f32)),
        .i64 => affon_metal_unary_chain_i64(try input.metalHandle(), try out.metalHandle(), &op_codes, &mins_i64, &maxs_i64, stages.len, out.bytes / @sizeOf(i64)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn binaryThenUnaryChain(dtype: DType, binary_tag: OpTag, lhs: *const Storage, rhs: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (stages.len == 0) return error.InvalidInputCount;
    if (stages.len > 16) return error.ExecutionNotImplemented;
    const binary_code = binaryCode(binary_tag) orelse return error.ExecutionNotImplemented;

    var op_codes: [16]u32 = [_]u32{0} ** 16;
    var mins_f32: [16]f32 = [_]f32{0} ** 16;
    var maxs_f32: [16]f32 = [_]f32{0} ** 16;
    var mins_i64: [16]i64 = [_]i64{0} ** 16;
    var maxs_i64: [16]i64 = [_]i64{0} ** 16;
    try encodeStages(stages, &op_codes, &mins_f32, &maxs_f32, &mins_i64, &maxs_i64);

    const rc = switch (dtype) {
        .f32 => affon_metal_binary_then_unary_chain_f32(try lhs.metalHandle(), try rhs.metalHandle(), try out.metalHandle(), binary_code, &op_codes, &mins_f32, &maxs_f32, stages.len, out.bytes / @sizeOf(f32)),
        .i64 => affon_metal_binary_then_unary_chain_i64(try lhs.metalHandle(), try rhs.metalHandle(), try out.metalHandle(), binary_code, &op_codes, &mins_i64, &maxs_i64, stages.len, out.bytes / @sizeOf(i64)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

fn encodeStages(stages: []const UnaryStage, op_codes: *[16]u32, mins_f32: *[16]f32, maxs_f32: *[16]f32, mins_i64: *[16]i64, maxs_i64: *[16]i64) !void {
    for (stages, 0..) |s, i| {
        op_codes[i] = unaryCode(s.tag) orelse return error.ExecutionNotImplemented;
        if (s.tag == .clamp) {
            const lo = s.clamp_min orelse return error.InvalidClampBounds;
            const hi = s.clamp_max orelse return error.InvalidClampBounds;
            mins_f32[i] = @floatCast(lo);
            maxs_f32[i] = @floatCast(hi);
            if (@trunc(lo) != lo or @trunc(hi) != hi) return error.InvalidClampBounds;
            mins_i64[i] = @intFromFloat(lo);
            maxs_i64[i] = @intFromFloat(hi);
        }
    }
}

fn unaryCode(tag: OpTag) ?u32 {
    return switch (tag) {
        .abs => 1,
        .neg => 2,
        .relu => 3,
        .sign => 4,
        .clamp => 5,
        .exp => 6,
        .log => 7,
        .sqrt => 8,
        .silu => 9,
        else => null,
    };
}

fn binaryCode(tag: OpTag) ?u32 {
    return switch (tag) {
        .add => 1,
        .sub => 2,
        .mul => 3,
        .div => 4,
        else => null,
    };
}
