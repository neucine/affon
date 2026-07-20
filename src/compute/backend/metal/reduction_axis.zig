const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const OpTag = @import("../../shared/types/operation/tag.zig").OpTag;
const common = @import("common.zig");

extern fn affon_metal_reduce_axis_sum_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_sum_i64(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_mean_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_min_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_min_i64(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_max_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_max_i64(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_variance_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_std_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_argmin_i64_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_argmax_i64_f32(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_argmin_i64_i64(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_argmax_i64_i64(a_handle: *anyopaque, out_handle: *anyopaque, rows: usize, cols: usize, axis: usize) c_int;
extern fn affon_metal_reduce_axis_nd_f32(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    reduce_kind: usize,
    out_len: usize,
) c_int;
extern fn affon_metal_reduce_axis_nd_i64(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    reduce_kind: usize,
    out_len: usize,
) c_int;
extern fn affon_metal_reduce_axis_nd_arg_i64_f32(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    choose_max: usize,
    out_len: usize,
) c_int;
extern fn affon_metal_reduce_axis_nd_arg_i64_i64(
    a_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    out_shape: [*]const u32,
    out_strides: [*]const u32,
    input_strides: [*]const u32,
    axis: usize,
    axis_size: usize,
    choose_max: usize,
    out_len: usize,
) c_int;

pub fn run(tag: OpTag, dtype: DType, input: *const Storage, out: *Storage, rows: usize, cols: usize, axis: usize, _: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;

    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => switch (tag) {
            .sum_axis => affon_metal_reduce_axis_sum_f32(input_handle, out_handle, rows, cols, axis),
            .mean_axis => affon_metal_reduce_axis_mean_f32(input_handle, out_handle, rows, cols, axis),
            .min_axis => affon_metal_reduce_axis_min_f32(input_handle, out_handle, rows, cols, axis),
            .max_axis => affon_metal_reduce_axis_max_f32(input_handle, out_handle, rows, cols, axis),
            .variance_axis => affon_metal_reduce_axis_variance_f32(input_handle, out_handle, rows, cols, axis),
            .std_axis => affon_metal_reduce_axis_std_f32(input_handle, out_handle, rows, cols, axis),
            .argmin_axis => affon_metal_reduce_axis_argmin_i64_f32(input_handle, out_handle, rows, cols, axis),
            .argmax_axis => affon_metal_reduce_axis_argmax_i64_f32(input_handle, out_handle, rows, cols, axis),
            else => return error.ExecutionNotImplemented,
        },
        .i64 => switch (tag) {
            .sum_axis => affon_metal_reduce_axis_sum_i64(input_handle, out_handle, rows, cols, axis),
            .min_axis => affon_metal_reduce_axis_min_i64(input_handle, out_handle, rows, cols, axis),
            .max_axis => affon_metal_reduce_axis_max_i64(input_handle, out_handle, rows, cols, axis),
            .argmin_axis => affon_metal_reduce_axis_argmin_i64_i64(input_handle, out_handle, rows, cols, axis),
            .argmax_axis => affon_metal_reduce_axis_argmax_i64_i64(input_handle, out_handle, rows, cols, axis),
            else => return error.ExecutionNotImplemented,
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runNd(tag: OpTag, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, out_elems: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (axis >= shape.len) return error.InvalidAxis;

    var out_shape_u32: [8]u32 = [_]u32{1} ** 8;
    var out_strides_u32: [8]u32 = [_]u32{1} ** 8;
    var input_strides_u32: [8]u32 = [_]u32{1} ** 8;
    for (shape, 0..) |d, i| out_shape_u32[i] = @intCast(d);
    out_shape_u32[axis] = 1;

    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        input_strides_u32[i] = @intCast(stride);
        stride *= shape[i];
    }

    stride = 1;
    i = shape.len;
    while (i > 0) {
        i -= 1;
        out_strides_u32[i] = @intCast(stride);
        stride *= @as(usize, out_shape_u32[i]);
    }

    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => switch (tag) {
            .sum_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 0, out_elems),
            .mean_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 1, out_elems),
            .min_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 2, out_elems),
            .max_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 3, out_elems),
            .variance_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 4, out_elems),
            .std_axis => affon_metal_reduce_axis_nd_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 5, out_elems),
            .argmin_axis => affon_metal_reduce_axis_nd_arg_i64_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 0, out_elems),
            .argmax_axis => affon_metal_reduce_axis_nd_arg_i64_f32(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 1, out_elems),
            else => return error.ExecutionNotImplemented,
        },
        .i64 => switch (tag) {
            .sum_axis => affon_metal_reduce_axis_nd_i64(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 0, out_elems),
            .min_axis => affon_metal_reduce_axis_nd_i64(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 2, out_elems),
            .max_axis => affon_metal_reduce_axis_nd_i64(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 3, out_elems),
            .argmin_axis => affon_metal_reduce_axis_nd_arg_i64_i64(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 0, out_elems),
            .argmax_axis => affon_metal_reduce_axis_nd_arg_i64_i64(input_handle, out_handle, shape.len, &out_shape_u32, &out_strides_u32, &input_strides_u32, axis, shape[axis], 1, out_elems),
            else => return error.ExecutionNotImplemented,
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
