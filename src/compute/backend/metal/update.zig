const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_muladd_inplace_f32(target_handle: *anyopaque, addend_handle: *anyopaque, scale: f32, len: usize) c_int;
extern fn affon_metal_axpy_inplace_f32(target_handle: *anyopaque, addend_handle: *anyopaque, scale: f32, len: usize) c_int;
extern fn affon_metal_sub_inplace_f32(target_handle: *anyopaque, delta_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_sub_inplace_i64(target_handle: *anyopaque, delta_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_scale_inplace_f32(target_handle: *anyopaque, scale: f32, len: usize) c_int;
extern fn affon_metal_adam_step_f32(
    param_handle: *anyopaque,
    m_handle: *anyopaque,
    v_handle: *anyopaque,
    grad_handle: *anyopaque,
    beta1: f32,
    beta2: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    eps: f32,
    lr: f32,
    weight_decay: f32,
    len: usize,
) c_int;

pub fn muladdInplace(dtype: DType, target: *Storage, addend: *const Storage, scale: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_muladd_inplace_f32(try target.metalHandle(), try addend.metalHandle(), @floatCast(scale), target.bytes / @sizeOf(f32)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn axpyInplace(dtype: DType, target: *Storage, addend: *const Storage, scale: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_axpy_inplace_f32(try target.metalHandle(), try addend.metalHandle(), @floatCast(scale), target.bytes / @sizeOf(f32)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn subInplace(dtype: DType, target: *Storage, delta: *const Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_sub_inplace_f32(try target.metalHandle(), try delta.metalHandle(), target.bytes / @sizeOf(f32)),
        .i64 => affon_metal_sub_inplace_i64(try target.metalHandle(), try delta.metalHandle(), target.bytes / @sizeOf(i64)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn scaleInplace(dtype: DType, target: *Storage, scale: f64) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rc = switch (dtype) {
        .f32 => affon_metal_scale_inplace_f32(try target.metalHandle(), @floatCast(scale), target.bytes / @sizeOf(f32)),
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}

pub fn adamStepInplace(
    dtype: DType,
    parameter: *Storage,
    first: *Storage,
    second: *Storage,
    gradient: *const Storage,
    beta1: f64,
    beta2: f64,
    bias_correction1: f64,
    bias_correction2: f64,
    eps: f64,
    lr: f64,
    weight_decay: f64,
) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    if (dtype != .f32) return error.ExecutionNotImplemented;
    const rc = affon_metal_adam_step_f32(
        try parameter.metalHandle(),
        try first.metalHandle(),
        try second.metalHandle(),
        try gradient.metalHandle(),
        @floatCast(beta1),
        @floatCast(beta2),
        @floatCast(bias_correction1),
        @floatCast(bias_correction2),
        @floatCast(eps),
        @floatCast(lr),
        @floatCast(weight_decay),
        parameter.bytes / @sizeOf(f32),
    );
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
