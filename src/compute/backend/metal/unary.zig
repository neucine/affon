const DType = @import("../../types/tensor/dtype.zig").DType;
const Storage = @import("../../types/tensor/storage.zig").Storage;
const OpTag = @import("../../types/operation/tag.zig").OpTag;
const common = @import("common.zig");
extern fn affon_metal_neg_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_abs_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_exp_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_log_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sqrt_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sign_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_relu_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sigmoid_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_silu_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_tanh_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_gelu_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_gelu_grad_f32(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_neg_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_abs_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sign_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_relu_i64(a_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;

pub fn run(tag: OpTag, dtype: DType, input: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => blk: {
            const len = out.bytes / @sizeOf(f32);
            break :blk switch (tag) {
                .abs => affon_metal_abs_f32(input_handle, out_handle, len),
                .exp => affon_metal_exp_f32(input_handle, out_handle, len),
                .log => affon_metal_log_f32(input_handle, out_handle, len),
                .neg => affon_metal_neg_f32(input_handle, out_handle, len),
                .sqrt => affon_metal_sqrt_f32(input_handle, out_handle, len),
                .sign => affon_metal_sign_f32(input_handle, out_handle, len),
                .relu => affon_metal_relu_f32(input_handle, out_handle, len),
                .sigmoid => affon_metal_sigmoid_f32(input_handle, out_handle, len),
                .silu => affon_metal_silu_f32(input_handle, out_handle, len),
                .tanh => affon_metal_tanh_f32(input_handle, out_handle, len),
                .gelu => affon_metal_gelu_f32(input_handle, out_handle, len),
                .gelu_grad => affon_metal_gelu_grad_f32(input_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        .i64 => blk: {
            const len = out.bytes / @sizeOf(i64);
            break :blk switch (tag) {
                .abs => affon_metal_abs_i64(input_handle, out_handle, len),
                .neg => affon_metal_neg_i64(input_handle, out_handle, len),
                .sign => affon_metal_sign_i64(input_handle, out_handle, len),
                .relu => affon_metal_relu_i64(input_handle, out_handle, len),
                else => return error.ExecutionNotImplemented,
            };
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
