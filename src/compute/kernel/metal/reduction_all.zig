const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const OpTag = @import("../../op/tag.zig").OpTag;
const common = @import("common.zig");
extern fn affon_metal_reduce_sum_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_sum_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_mean_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_min_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_min_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_max_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_max_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_argmin_i64_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_argmax_i64_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_argmin_i64_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_argmax_i64_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_variance_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_reduce_std_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;

pub fn run(tag: OpTag, dtype: DType, input: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;

    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => blk: {
            const len = input.bytes / @sizeOf(f32);
            break :blk switch (tag) {
                .sum_all => affon_metal_reduce_sum_f32(input_handle, out_handle, len),
                .mean_all => affon_metal_reduce_mean_f32(input_handle, out_handle, len),
                .min_all => affon_metal_reduce_min_f32(input_handle, out_handle, len),
                .max_all => affon_metal_reduce_max_f32(input_handle, out_handle, len),
                .argmin_all => affon_metal_reduce_argmin_i64_f32(input_handle, out_handle, len),
                .argmax_all => affon_metal_reduce_argmax_i64_f32(input_handle, out_handle, len),
                .variance_all => affon_metal_reduce_variance_f32(input_handle, out_handle, len),
                .std_all => affon_metal_reduce_std_f32(input_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        .i64 => blk: {
            const len = input.bytes / @sizeOf(i64);
            break :blk switch (tag) {
                .sum_all => affon_metal_reduce_sum_i64(input_handle, out_handle, len),
                .min_all => affon_metal_reduce_min_i64(input_handle, out_handle, len),
                .max_all => affon_metal_reduce_max_i64(input_handle, out_handle, len),
                .argmin_all => affon_metal_reduce_argmin_i64_i64(input_handle, out_handle, len),
                .argmax_all => affon_metal_reduce_argmax_i64_i64(input_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
