const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const SliceRange = @import("../../shared/types/operation/options.zig").SliceRange;
const common = @import("common.zig");

extern fn affon_metal_contiguous_f32(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const u32,
    offset: usize,
    len: usize,
) c_int;
extern fn affon_metal_contiguous_i64(
    input_handle: *anyopaque,
    out_handle: *anyopaque,
    ndim: usize,
    shape: [*]const u32,
    strides: [*]const u32,
    offset: usize,
    len: usize,
) c_int;

pub fn run(dtype: DType, input: *const Storage, out: *Storage, in_shape: []const usize, ranges: []const SliceRange) !void {
    if (!common.isAvailable()) return error.MetalUnavailable;
    const rank = in_shape.len;
    if (rank > 8) return error.ExecutionNotImplemented;

    var starts: [8]usize = [_]usize{0} ** 8;
    var steps: [8]usize = [_]usize{1} ** 8;
    var out_dims: [8]usize = [_]usize{1} ** 8;
    var in_strides: [8]usize = [_]usize{1} ** 8;
    var view_strides: [8]u32 = [_]u32{1} ** 8;
    var shape_u32: [8]u32 = [_]u32{1} ** 8;

    var i: usize = 0;
    while (i < rank) : (i += 1) {
        const r = if (i < ranges.len) ranges[i] else SliceRange{ .start = 0, .stop = in_shape[i], .step = 1 };
        if (r.step <= 0) return error.InvalidSliceStep;
        if (r.start > r.stop or r.stop > in_shape[i]) return error.ShapeMismatch;

        starts[i] = r.start;
        steps[i] = @intCast(r.step);
        out_dims[i] = if (r.start >= r.stop) 0 else ((r.stop - r.start - 1) / steps[i]) + 1;
        shape_u32[i] = @intCast(out_dims[i]);
    }

    if (rank > 0) {
        in_strides[rank - 1] = 1;
        var d: usize = rank - 1;
        while (d > 0) {
            d -= 1;
            in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        }
    }

    var offset: usize = 0;
    var out_elems: usize = if (rank == 0) 1 else 1;
    i = 0;
    while (i < rank) : (i += 1) {
        offset += starts[i] * in_strides[i];
        const vstride = in_strides[i] * steps[i];
        view_strides[i] = @intCast(vstride);
        out_elems *= out_dims[i];
    }

    const input_handle = try input.metalHandle();
    const out_handle = try out.metalHandle();
    const rc = switch (dtype) {
        .f32 => blk: {
            if (out.bytes != out_elems * @sizeOf(f32)) return error.ShapeMismatch;
            break :blk affon_metal_contiguous_f32(
                input_handle,
                out_handle,
                rank,
                &shape_u32,
                &view_strides,
                offset,
                out_elems,
            );
        },
        .i64 => blk: {
            if (out.bytes != out_elems * @sizeOf(i64)) return error.ShapeMismatch;
            break :blk affon_metal_contiguous_i64(
                input_handle,
                out_handle,
                rank,
                &shape_u32,
                &view_strides,
                offset,
                out_elems,
            );
        },
        else => return error.ExecutionNotImplemented,
    };
    if (rc != 0) return error.MetalKernelLaunchFailed;
}
