const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_masked_fill_i64_f32(input_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, fill_value: f32, len: usize) c_int;
extern fn affon_metal_masked_fill_i64_i64(input_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, fill_value: i64, len: usize) c_int;
extern fn affon_metal_masked_fill_broadcast_i64_f32(input_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, mask_strides: [*]const u32, input_offset: usize, mask_offset: usize, fill_value: f32, len: usize) c_int;
extern fn affon_metal_masked_fill_broadcast_i64_i64(input_handle: *anyopaque, mask_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, input_strides: [*]const u32, mask_strides: [*]const u32, input_offset: usize, mask_offset: usize, fill_value: i64, len: usize) c_int;

pub fn run(input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, value: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (mask_dtype != .i64) return error.ExecutionNotImplemented;

    const rc = switch (input_dtype) {
        .f32 => affon_metal_masked_fill_i64_f32(
            try input.metalHandle(),
            try mask.metalHandle(),
            try out.metalHandle(),
            @floatCast(value),
            out.bytes / @sizeOf(f32),
        ),
        .i64 => affon_metal_masked_fill_i64_i64(
            try input.metalHandle(),
            try mask.metalHandle(),
            try out.metalHandle(),
            @intFromFloat(value),
            out.bytes / @sizeOf(i64),
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runBroadcast(input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, shape: []const usize, input_strides: []const isize, mask_strides: []const isize, input_offset: usize, mask_offset: usize, value: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (mask_dtype != .i64) return error.ExecutionNotImplemented;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (input_strides.len != shape.len or mask_strides.len != shape.len) return error.ExecutionNotImplemented;

    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var input_u32: [8]u32 = [_]u32{0} ** 8;
    var mask_u32: [8]u32 = [_]u32{0} ** 8;
    var len: usize = 1;
    for (shape, 0..) |dim, i| {
        shape_u32[i] = @intCast(dim);
        len *= dim;
        if (input_strides[i] < 0 or mask_strides[i] < 0) return error.ExecutionNotImplemented;
        input_u32[i] = @intCast(input_strides[i]);
        mask_u32[i] = @intCast(mask_strides[i]);
    }

    const rc = switch (input_dtype) {
        .f32 => affon_metal_masked_fill_broadcast_i64_f32(
            try input.metalHandle(),
            try mask.metalHandle(),
            try out.metalHandle(),
            shape.len,
            &shape_u32,
            &input_u32,
            &mask_u32,
            input_offset,
            mask_offset,
            @floatCast(value),
            len,
        ),
        .i64 => affon_metal_masked_fill_broadcast_i64_i64(
            try input.metalHandle(),
            try mask.metalHandle(),
            try out.metalHandle(),
            shape.len,
            &shape_u32,
            &input_u32,
            &mask_u32,
            input_offset,
            mask_offset,
            @intFromFloat(value),
            len,
        ),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
