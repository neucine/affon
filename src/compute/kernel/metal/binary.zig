const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const OpTag = @import("../../op/tag.zig").OpTag;
const common = @import("common.zig");
extern fn affon_metal_add_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sub_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_mul_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_div_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_eq_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_lt_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_gt_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_add_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sub_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_mul_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_div_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_eq_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_lt_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_gt_i64(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_add_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_sub_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_mul_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_div_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_eq_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_lt_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;
extern fn affon_metal_gt_broadcast_f32(a_handle: *anyopaque, b_handle: *anyopaque, out_handle: *anyopaque, ndim: usize, shape: [*]const u32, a_strides: [*]const u32, b_strides: [*]const u32, a_offset: usize, b_offset: usize, len: usize) c_int;

pub fn run(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const a_handle = try a.metalHandle();
    const b_handle = try b.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => blk: {
            const len = out.bytes / @sizeOf(f32);
            break :blk switch (tag) {
                .add => affon_metal_add_f32(a_handle, b_handle, out_handle, len),
                .sub => affon_metal_sub_f32(a_handle, b_handle, out_handle, len),
                .mul => affon_metal_mul_f32(a_handle, b_handle, out_handle, len),
                .div => affon_metal_div_f32(a_handle, b_handle, out_handle, len),
                .eq => affon_metal_eq_f32(a_handle, b_handle, out_handle, len),
                .lt => affon_metal_lt_f32(a_handle, b_handle, out_handle, len),
                .gt => affon_metal_gt_f32(a_handle, b_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        .i64 => blk: {
            const len = out.bytes / @sizeOf(i64);
            break :blk switch (tag) {
                .add => affon_metal_add_i64(a_handle, b_handle, out_handle, len),
                .sub => affon_metal_sub_i64(a_handle, b_handle, out_handle, len),
                .mul => affon_metal_mul_i64(a_handle, b_handle, out_handle, len),
                .div => affon_metal_div_i64(a_handle, b_handle, out_handle, len),
                .eq => affon_metal_eq_i64(a_handle, b_handle, out_handle, len),
                .lt => affon_metal_lt_i64(a_handle, b_handle, out_handle, len),
                .gt => affon_metal_gt_i64(a_handle, b_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn runBroadcast(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    if (shape.len == 0 or shape.len > 8) return error.ExecutionNotImplemented;
    if (a_strides.len != shape.len or b_strides.len != shape.len) return error.ExecutionNotImplemented;

    var shape_u32: [8]u32 = [_]u32{1} ** 8;
    var a_strides_u32: [8]u32 = [_]u32{0} ** 8;
    var b_strides_u32: [8]u32 = [_]u32{0} ** 8;
    for (shape, 0..) |d, i| shape_u32[i] = @intCast(d);
    for (a_strides, 0..) |s, i| {
        if (s < 0) return error.ExecutionNotImplemented;
        a_strides_u32[i] = @intCast(s);
    }
    for (b_strides, 0..) |s, i| {
        if (s < 0) return error.ExecutionNotImplemented;
        b_strides_u32[i] = @intCast(s);
    }

    var len: usize = 1;
    for (shape) |d| len *= d;
    const a_handle = try a.metalHandle();
    const b_handle = try b.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (tag) {
        .add => affon_metal_add_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .sub => affon_metal_sub_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .mul => affon_metal_mul_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .div => affon_metal_div_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .eq => affon_metal_eq_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .lt => affon_metal_lt_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        .gt => affon_metal_gt_broadcast_f32(a_handle, b_handle, out_handle, shape.len, &shape_u32, &a_strides_u32, &b_strides_u32, a_offset, b_offset, len),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn compare(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    return run(tag, dtype, a, b, out);
}

pub fn compareBroadcast(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    return runBroadcast(tag, dtype, a, b, out, shape, a_strides, b_strides, a_offset, b_offset);
}
