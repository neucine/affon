const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const common = @import("common.zig");

extern fn affon_metal_cast_f32_f32(input_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_cast_i64_i64(input_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_cast_f32_i64(input_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;
extern fn affon_metal_cast_i64_f32(input_handle: *anyopaque, out_handle: *anyopaque, len: usize) c_int;

pub fn run(from: DType, to: DType, input: *const Storage, out: *Storage) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const in_h = try input.metalHandle();
    const out_h = try out.metalHandle();
    const rc = switch (from) {
        .f32 => switch (to) {
            .f32 => affon_metal_cast_f32_f32(in_h, out_h, out.bytes / @sizeOf(f32)),
            .i64 => affon_metal_cast_f32_i64(in_h, out_h, out.bytes / @sizeOf(i64)),
            else => return error.ExecutionNotImplemented,
        },
        .i64 => switch (to) {
            .i64 => affon_metal_cast_i64_i64(in_h, out_h, out.bytes / @sizeOf(i64)),
            .f32 => affon_metal_cast_i64_f32(in_h, out_h, out.bytes / @sizeOf(f32)),
            else => return error.ExecutionNotImplemented,
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
