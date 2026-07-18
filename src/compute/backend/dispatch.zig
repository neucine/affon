const Device = @import("../types/tensor/device.zig").Device;
const DType = @import("../types/tensor/dtype.zig").DType;
const Layout = @import("../types/tensor/layout.zig").Layout;
const Storage = @import("../types/tensor/storage.zig").Storage;
const std = @import("std");
const OpTag = @import("../types/operation/tag.zig").OpTag;
const cpu = @import("cpu/index.zig");
const metal = @import("metal/index.zig");

fn broadcastIndex(flat: usize, out_shape: []const usize, in_strides: []const isize, in_offset: usize) usize {
    var rem = flat;
    var idx: usize = in_offset;
    var axis: usize = out_shape.len;
    while (axis > 0) {
        axis -= 1;
        const dim = out_shape[axis];
        const coord = if (dim == 0) 0 else rem % dim;
        rem = if (dim == 0) 0 else rem / dim;
        const stride = in_strides[axis];
        if (stride != 0) idx += coord * @as(usize, @intCast(stride));
    }
    return idx;
}

fn maskAt(dtype: DType, bytes: []align(8) const u8, i: usize) bool {
    return switch (dtype) {
        .f32 => std.mem.bytesAsSlice(f32, bytes)[i] != 0,
        .f64 => std.mem.bytesAsSlice(f64, bytes)[i] != 0,
        .i64 => std.mem.bytesAsSlice(i64, bytes)[i] != 0,
    };
}

const MatmulOperand = struct {
    shape: [8]usize = [_]usize{1} ** 8,
    strides: [8]isize = [_]isize{0} ** 8,
    rank: usize,
    batch_rank: usize,
    offset: usize,
    row_stride: isize,
    col_stride: isize,
    m: usize,
    k: usize,
    n: usize,
};

fn normalizeMatmulOperand(shape_in: []const usize, layout: Layout, side: enum { lhs, rhs }) !MatmulOperand {
    var operand = MatmulOperand{
        .rank = 0,
        .batch_rank = 0,
        .offset = layout.offset,
        .row_stride = 0,
        .col_stride = 0,
        .m = 0,
        .k = 0,
        .n = 0,
    };
    if (shape_in.len == 0 or shape_in.len > 8) return error.ExecutionNotImplemented;
    if (shape_in.len == 1) {
        operand.rank = 2;
        operand.batch_rank = 0;
        operand.shape[0] = if (side == .lhs) 1 else shape_in[0];
        operand.shape[1] = if (side == .lhs) shape_in[0] else 1;
        if (side == .lhs) {
            operand.strides[0] = 0;
            operand.strides[1] = layout.strides[0];
        } else {
            operand.strides[0] = layout.strides[0];
            operand.strides[1] = 0;
        }
    } else {
        operand.rank = shape_in.len;
        operand.batch_rank = shape_in.len - 2;
        @memcpy(operand.shape[0..shape_in.len], shape_in);
        @memcpy(operand.strides[0..shape_in.len], layout.strides);
    }
    operand.m = operand.shape[operand.rank - 2];
    operand.k = operand.shape[operand.rank - 1];
    operand.row_stride = operand.strides[operand.rank - 2];
    operand.col_stride = operand.strides[operand.rank - 1];
    operand.n = if (side == .rhs) operand.shape[operand.rank - 1] else 0;
    return operand;
}

fn computeBroadcastBatchOffset(batch_rank: usize, batch_coord: []const usize, operand_batch_rank: usize, shape: []const usize, strides: []const isize) usize {
    var offset: usize = 0;
    const axis_offset = batch_rank - operand_batch_rank;
    for (0..operand_batch_rank) |local_axis| {
        const coord = if (shape[local_axis] == 1) 0 else batch_coord[axis_offset + local_axis];
        offset += coord * @as(usize, @intCast(strides[local_axis]));
    }
    return offset;
}

fn cpuBinaryBroadcast(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    const out_bytes = try out.writableBytes();
    const len = out.bytes / dtype.size();
    switch (dtype) {
        .f32 => {
            const lhs = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(f32, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, out_bytes);
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .add => lhs[ai] + rhs[bi],
                    .sub => lhs[ai] - rhs[bi],
                    .mul => lhs[ai] * rhs[bi],
                    .div => lhs[ai] / rhs[bi],
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
        .f64 => {
            const lhs = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(f64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, out_bytes);
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .add => lhs[ai] + rhs[bi],
                    .sub => lhs[ai] - rhs[bi],
                    .mul => lhs[ai] * rhs[bi],
                    .div => lhs[ai] / rhs[bi],
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
        .i64 => {
            const lhs = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(i64, try b.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, out_bytes);
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .add => lhs[ai] + rhs[bi],
                    .sub => lhs[ai] - rhs[bi],
                    .mul => lhs[ai] * rhs[bi],
                    .div => @divTrunc(lhs[ai], rhs[bi]),
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
    }
}

fn cpuCompareBroadcast(tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    const out_bytes = try out.writableBytes();
    const len = out.bytes / @sizeOf(i64);
    const dst = std.mem.bytesAsSlice(i64, out_bytes);
    switch (dtype) {
        .f32 => {
            const lhs = std.mem.bytesAsSlice(f32, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(f32, try b.readableBytes());
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .eq => @intFromBool(lhs[ai] == rhs[bi]),
                    .lt => @intFromBool(lhs[ai] < rhs[bi]),
                    .gt => @intFromBool(lhs[ai] > rhs[bi]),
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
        .f64 => {
            const lhs = std.mem.bytesAsSlice(f64, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(f64, try b.readableBytes());
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .eq => @intFromBool(lhs[ai] == rhs[bi]),
                    .lt => @intFromBool(lhs[ai] < rhs[bi]),
                    .gt => @intFromBool(lhs[ai] > rhs[bi]),
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
        .i64 => {
            const lhs = std.mem.bytesAsSlice(i64, try a.readableBytes());
            const rhs = std.mem.bytesAsSlice(i64, try b.readableBytes());
            for (0..len) |flat| {
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = switch (tag) {
                    .eq => @intFromBool(lhs[ai] == rhs[bi]),
                    .lt => @intFromBool(lhs[ai] < rhs[bi]),
                    .gt => @intFromBool(lhs[ai] > rhs[bi]),
                    else => return error.ExecutionNotImplemented,
                };
            }
        },
    }
}

fn cpuWhereBroadcast(cond_dtype: DType, value_dtype: DType, cond: *const Storage, on_true: *const Storage, on_false: *const Storage, out: *Storage, shape: []const usize, cond_strides: []const isize, a_strides: []const isize, b_strides: []const isize, cond_offset: usize, a_offset: usize, b_offset: usize) !void {
    const cond_bytes = try cond.readableBytes();
    const out_bytes = try out.writableBytes();
    const len = out.bytes / value_dtype.size();
    switch (value_dtype) {
        .f32 => {
            const a = std.mem.bytesAsSlice(f32, try on_true.readableBytes());
            const b = std.mem.bytesAsSlice(f32, try on_false.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, out_bytes);
            for (0..len) |flat| {
                const ci = broadcastIndex(flat, shape, cond_strides, cond_offset);
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = if (maskAt(cond_dtype, cond_bytes, ci)) a[ai] else b[bi];
            }
        },
        .f64 => {
            const a = std.mem.bytesAsSlice(f64, try on_true.readableBytes());
            const b = std.mem.bytesAsSlice(f64, try on_false.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, out_bytes);
            for (0..len) |flat| {
                const ci = broadcastIndex(flat, shape, cond_strides, cond_offset);
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = if (maskAt(cond_dtype, cond_bytes, ci)) a[ai] else b[bi];
            }
        },
        .i64 => {
            const a = std.mem.bytesAsSlice(i64, try on_true.readableBytes());
            const b = std.mem.bytesAsSlice(i64, try on_false.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, out_bytes);
            for (0..len) |flat| {
                const ci = broadcastIndex(flat, shape, cond_strides, cond_offset);
                const ai = broadcastIndex(flat, shape, a_strides, a_offset);
                const bi = broadcastIndex(flat, shape, b_strides, b_offset);
                dst[flat] = if (maskAt(cond_dtype, cond_bytes, ci)) a[ai] else b[bi];
            }
        },
    }
}

fn cpuMaskedFillBroadcast(input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, shape: []const usize, input_strides: []const isize, mask_strides: []const isize, input_offset: usize, mask_offset: usize, value: f64) !void {
    const mask_bytes = try mask.readableBytes();
    const out_bytes = try out.writableBytes();
    const len = out.bytes / input_dtype.size();
    switch (input_dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, out_bytes);
            const fill: f32 = @floatCast(value);
            for (0..len) |flat| {
                const ii = broadcastIndex(flat, shape, input_strides, input_offset);
                const mi = broadcastIndex(flat, shape, mask_strides, mask_offset);
                dst[flat] = if (maskAt(mask_dtype, mask_bytes, mi)) fill else src[ii];
            }
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, out_bytes);
            for (0..len) |flat| {
                const ii = broadcastIndex(flat, shape, input_strides, input_offset);
                const mi = broadcastIndex(flat, shape, mask_strides, mask_offset);
                dst[flat] = if (maskAt(mask_dtype, mask_bytes, mi)) value else src[ii];
            }
        },
        .i64 => {
            const src = std.mem.bytesAsSlice(i64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(i64, out_bytes);
            const fill: i64 = @intFromFloat(value);
            for (0..len) |flat| {
                const ii = broadcastIndex(flat, shape, input_strides, input_offset);
                const mi = broadcastIndex(flat, shape, mask_strides, mask_offset);
                dst[flat] = if (maskAt(mask_dtype, mask_bytes, mi)) fill else src[ii];
            }
        },
    }
}

pub const UnaryStage = struct {
    tag: OpTag,
    clamp_min: ?f64 = null,
    clamp_max: ?f64 = null,
};

pub const UpdateTag = enum {
    muladd,
    axpy,
    sub,
    scale,
};

pub fn copyStorage(device: Device, src: *const Storage, dst: *Storage, byte_len: usize) !void {
    if (src.device() != device or dst.device() != device) return error.DeviceMismatch;
    if (byte_len > src.bytes or byte_len > dst.bytes) return error.SizeMismatch;
    switch (device) {
        .cpu => {
            const src_bytes = try src.readableBytes();
            const dst_bytes = try dst.writableBytes();
            @memcpy(dst_bytes[0..byte_len], src_bytes[0..byte_len]);
        },
        .metal => try metal.copy(src, dst, byte_len),
    }
}

pub fn binary(device: Device, tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => switch (tag) {
            .add => try cpu.add(dtype, a, b, out),
            .sub => try cpu.sub(dtype, a, b, out),
            .mul => try cpu.mul(dtype, a, b, out),
            .div => try cpu.div(dtype, a, b, out),
            else => return error.ExecutionNotImplemented,
        },
        .metal => try metal.binary(tag, dtype, a, b, out),
    }
}

pub fn compare(device: Device, tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => switch (tag) {
            .eq => try cpu.eq(dtype, a, b, out),
            .lt => try cpu.lt(dtype, a, b, out),
            .gt => try cpu.gt(dtype, a, b, out),
            else => return error.ExecutionNotImplemented,
        },
        .metal => try metal.compare(tag, dtype, a, b, out),
    }
}

pub fn binaryBroadcast(device: Device, tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    switch (device) {
        .cpu => try cpuBinaryBroadcast(tag, dtype, a, b, out, shape, a_strides, b_strides, a_offset, b_offset),
        .metal => try metal.binary_broadcast(tag, dtype, a, b, out, shape, a_strides, b_strides, a_offset, b_offset),
    }
}

pub fn compareBroadcast(device: Device, tag: OpTag, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, a_strides: []const isize, b_strides: []const isize, a_offset: usize, b_offset: usize) !void {
    switch (device) {
        .cpu => try cpuCompareBroadcast(tag, dtype, a, b, out, shape, a_strides, b_strides, a_offset, b_offset),
        .metal => try metal.compare_broadcast(tag, dtype, a, b, out, shape, a_strides, b_strides, a_offset, b_offset),
    }
}

pub fn unary(device: Device, tag: OpTag, dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => switch (tag) {
            .abs => try cpu.abs(dtype, input, out),
            .exp => try cpu.exp(dtype, input, out),
            .gelu => try cpu.gelu(dtype, input, out),
            .gelu_grad => try cpu.gelu_grad(dtype, input, out),
            .log => try cpu.log(dtype, input, out),
            .neg => try cpu.neg(dtype, input, out),
            .sqrt => try cpu.sqrt(dtype, input, out),
            .sign => try cpu.sign(dtype, input, out),
            .relu => try cpu.relu(dtype, input, out),
            .sigmoid => try cpu.sigmoid(dtype, input, out),
            .silu => try cpu.silu(dtype, input, out),
            .tanh => try cpu.tanh(dtype, input, out),
            else => return error.ExecutionNotImplemented,
        },
        .metal => try metal.unary(tag, dtype, input, out),
    }
}

pub fn unaryClamp(device: Device, dtype: DType, input: *const Storage, out: *Storage, min_val: f64, max_val: f64) !void {
    switch (device) {
        .cpu => try cpu.clamp(dtype, input, out, min_val, max_val),
        .metal => try metal.clamp(dtype, input, out, min_val, max_val),
    }
}

pub fn update(device: Device, tag: UpdateTag, dtype: DType, target: *Storage, other: ?*const Storage, scalar: ?f64) !void {
    switch (device) {
        .cpu => switch (tag) {
            .muladd => try cpu.update.muladdInplace(dtype, target, other orelse return error.InvalidArgument, scalar orelse return error.InvalidArgument),
            .axpy => try cpu.update.axpyInplace(dtype, target, other orelse return error.InvalidArgument, scalar orelse return error.InvalidArgument),
            .sub => try cpu.update.subInplace(dtype, target, other orelse return error.InvalidArgument),
            .scale => try cpu.update.scaleInplace(dtype, target, scalar orelse return error.InvalidArgument),
        },
        .metal => switch (tag) {
            .muladd => try metal.update.muladdInplace(dtype, target, other orelse return error.InvalidArgument, scalar orelse return error.InvalidArgument),
            .axpy => try metal.update.axpyInplace(dtype, target, other orelse return error.InvalidArgument, scalar orelse return error.InvalidArgument),
            .sub => try metal.update.subInplace(dtype, target, other orelse return error.InvalidArgument),
            .scale => try metal.update.scaleInplace(dtype, target, scalar orelse return error.InvalidArgument),
        },
    }
}

pub fn where(device: Device, cond_dtype: DType, value_dtype: DType, cond: *const Storage, on_true: *const Storage, on_false: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => try cpu.where(cond_dtype, value_dtype, cond, on_true, on_false, out),
        .metal => try metal.where(cond_dtype, value_dtype, cond, on_true, on_false, out),
    }
}

pub fn whereBroadcastF32(
    device: Device,
    cond: *const Storage,
    on_true: *const Storage,
    on_false: *const Storage,
    out: *Storage,
    shape: []const usize,
    cond_strides: []const isize,
    a_strides: []const isize,
    b_strides: []const isize,
    cond_offset: usize,
    a_offset: usize,
    b_offset: usize,
) !void {
    switch (device) {
        .cpu => return error.ExecutionNotImplemented,
        .metal => try metal.where_broadcast_f32(cond, on_true, on_false, out, shape, cond_strides, a_strides, b_strides, cond_offset, a_offset, b_offset),
    }
}

pub fn whereBroadcast(device: Device, cond_dtype: DType, value_dtype: DType, cond: *const Storage, on_true: *const Storage, on_false: *const Storage, out: *Storage, shape: []const usize, cond_strides: []const isize, a_strides: []const isize, b_strides: []const isize, cond_offset: usize, a_offset: usize, b_offset: usize) !void {
    switch (device) {
        .cpu => try cpuWhereBroadcast(cond_dtype, value_dtype, cond, on_true, on_false, out, shape, cond_strides, a_strides, b_strides, cond_offset, a_offset, b_offset),
        .metal => {
            if (value_dtype == .f32 and (cond_dtype == .f32 or cond_dtype == .i64)) {
                try metal.where_broadcast_f32(cond_dtype, cond, on_true, on_false, out, shape, cond_strides, a_strides, b_strides, cond_offset, a_offset, b_offset);
                return;
            }
            return error.ExecutionNotImplemented;
        },
    }
}

pub fn maskedFill(device: Device, input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, value: f64) !void {
    switch (device) {
        .cpu => try cpu.masked_fill(input_dtype, mask_dtype, input, mask, out, value),
        .metal => try metal.masked_fill(input_dtype, mask_dtype, input, mask, out, value),
    }
}

pub fn maskedFillBroadcast(device: Device, input_dtype: DType, mask_dtype: DType, input: *const Storage, mask: *const Storage, out: *Storage, shape: []const usize, input_strides: []const isize, mask_strides: []const isize, input_offset: usize, mask_offset: usize, value: f64) !void {
    switch (device) {
        .cpu => try cpuMaskedFillBroadcast(input_dtype, mask_dtype, input, mask, out, shape, input_strides, mask_strides, input_offset, mask_offset, value),
        .metal => try metal.masked_fill_broadcast(input_dtype, mask_dtype, input, mask, out, shape, input_strides, mask_strides, input_offset, mask_offset, value),
    }
}

pub fn cast(device: Device, from: DType, to: DType, input: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => try cpu.cast(from, to, input, out),
        .metal => try metal.cast(from, to, input, out),
    }
}

pub fn reductionAll(device: Device, tag: OpTag, dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => switch (tag) {
            .sum_all => try cpu.sum_all(dtype, input, out),
            .mean_all => try cpu.mean_all(dtype, input, out),
            .min_all => try cpu.min_all(dtype, input, out),
            .max_all => try cpu.max_all(dtype, input, out),
            .argmin_all => try cpu.argmin_all(dtype, input, out),
            .argmax_all => try cpu.argmax_all(dtype, input, out),
            .variance_all => try cpu.variance_all(dtype, input, out),
            .std_all => try cpu.std_all(dtype, input, out),
            else => return error.ExecutionNotImplemented,
        },
        .metal => try metal.reduction_all(tag, dtype, input, out),
    }
}

pub fn softmax(device: Device, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, layout: Layout, axis: usize) !void {
    switch (device) {
        .cpu => try cpu.softmax(dtype, input, out, shape, axis),
        .metal => try metal.softmax(dtype, input, out, shape, layout, axis),
    }
}

pub fn logSoftmax(allocator: std.mem.Allocator, device: Device, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, layout: Layout, axis: usize) !void {
    const bytes = out.bytes;
    const temp = switch (device) {
        .cpu => try Storage.createCpu(allocator, bytes, false),
        .metal => try Storage.createMetalWithMetadata(allocator, bytes, .{
            .policy = .pooled,
            .reason = .op_output,
            .source = .eager,
        }),
    };
    defer temp.release();
    try softmax(device, dtype, input, temp, shape, layout, axis);
    try unary(device, .log, dtype, temp, out);
}

pub fn logSoftmaxNll(device: Device, dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    switch (device) {
        .cpu => try cpu.log_softmax_nll(dtype, logits, targets, out, shape, axis),
        .metal => try metal.log_softmax_nll(dtype, logits, targets, out, shape, axis),
    }
}

pub fn crossEntropyIndexed(device: Device, dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, rows: usize, classes: usize) !void {
    switch (device) {
        .cpu => try cpu.cross_entropy_indexed(dtype, logits, targets, out, rows, classes),
        .metal => try metal.cross_entropy_indexed(dtype, logits, targets, out, rows, classes),
    }
}

pub fn crossEntropyIndexedTransposed(device: Device, dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, rows: usize, classes: usize) !void {
    switch (device) {
        .cpu => try cpu.cross_entropy_indexed_transposed(dtype, logits, targets, out, rows, classes),
        .metal => try metal.cross_entropy_indexed_transposed(dtype, logits, targets, out, rows, classes),
    }
}

pub fn crossEntropyIndexedBackward(device: Device, dtype: DType, logits: *const Storage, targets: *const Storage, grad_out: *const Storage, out: *Storage, rows: usize, classes: usize) !void {
    switch (device) {
        .cpu => try cpu.cross_entropy_indexed_backward(dtype, logits, targets, grad_out, out, rows, classes),
        .metal => try metal.cross_entropy_indexed_backward(dtype, logits, targets, grad_out, out, rows, classes),
    }
}

pub fn addLayerNorm(device: Device, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
    switch (device) {
        .cpu => try cpu.add_layer_norm(dtype, a, b, out, shape, axis, eps),
        .metal => try metal.add_layer_norm(dtype, a, b, out, shape, axis, eps),
    }
}

pub fn attentionScores(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    q: *const Storage,
    k_t: *const Storage,
    scale: *const Storage,
    mask: *const Storage,
    out: *Storage,
    q_shape: []const usize,
    k_t_shape: []const usize,
    out_shape: []const usize,
    softmax_axis: usize,
    mask_fill_value: f64,
) !void {
    if (out_shape.len == 0 or softmax_axis >= out_shape.len) return error.InvalidAxis;
    if (softmax_axis + 1 != out_shape.len) return error.ExecutionNotImplemented;
    if (dtype == .i64) return error.ExecutionNotImplemented;
    if (!std.mem.eql(usize, q_shape[0 .. q_shape.len - 2], out_shape[0 .. out_shape.len - 2])) return error.ShapeMismatch;

    var aa = q_shape;
    var bb = k_t_shape;
    if (aa.len == 1) aa = &.{ 1, aa[0] };
    if (bb.len == 1) bb = &.{ bb[0], 1 };
    if (aa.len < 2 or bb.len < 2) return error.ExecutionNotImplemented;
    if (aa[aa.len - 1] != bb[bb.len - 2]) return error.ShapeMismatch;

    const a_batch = aa[0 .. aa.len - 2];
    const b_batch = bb[0 .. bb.len - 2];
    const batch_rank = @max(a_batch.len, b_batch.len);
    if (batch_rank > 8) return error.ExecutionNotImplemented;

    var out_batch_shape: [8]usize = [_]usize{1} ** 8;
    var a_batch_shape: [8]usize = [_]usize{1} ** 8;
    var b_batch_shape: [8]usize = [_]usize{1} ** 8;
    for (0..batch_rank) |i| {
        const ai = if (i + a_batch.len >= batch_rank) a_batch[i + a_batch.len - batch_rank] else 1;
        const bi = if (i + b_batch.len >= batch_rank) b_batch[i + b_batch.len - batch_rank] else 1;
        if (ai != bi and ai != 1 and bi != 1) return error.ExecutionNotImplemented;
        out_batch_shape[i] = @max(ai, bi);
        a_batch_shape[i] = ai;
        b_batch_shape[i] = bi;
    }

    var out_batch_strides: [8]usize = [_]usize{1} ** 8;
    var a_batch_strides: [8]usize = [_]usize{1} ** 8;
    var b_batch_strides: [8]usize = [_]usize{1} ** 8;
    if (batch_rank > 0) {
        out_batch_strides[batch_rank - 1] = 1;
        a_batch_strides[batch_rank - 1] = 1;
        b_batch_strides[batch_rank - 1] = 1;
        var d = batch_rank - 1;
        while (d > 0) {
            d -= 1;
            out_batch_strides[d] = out_batch_strides[d + 1] * out_batch_shape[d + 1];
            a_batch_strides[d] = a_batch_strides[d + 1] * a_batch_shape[d + 1];
            b_batch_strides[d] = b_batch_strides[d + 1] * b_batch_shape[d + 1];
        }
    }

    var batch: usize = 1;
    for (out_batch_shape[0..batch_rank]) |d| batch *= d;
    const m = aa[aa.len - 2];
    const k = aa[aa.len - 1];
    const n = bb[bb.len - 1];
    const elem_bytes: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => unreachable,
    };
    const a_batch_bytes = m * k * elem_bytes;
    const b_batch_bytes = k * n * elem_bytes;
    const out_batch_bytes = m * n * elem_bytes;
    const scale_batch_bytes = m * n * elem_bytes;
    const mask_batch_bytes = m * n * @sizeOf(i64);

    var coord: [8]usize = [_]usize{0} ** 8;
    for (0..batch) |bi| {
        var rem = bi;
        for (0..batch_rank) |r| {
            const d = batch_rank - 1 - r;
            coord[d] = rem % out_batch_shape[d];
            rem /= out_batch_shape[d];
        }

        var a_batch_idx: usize = 0;
        var b_batch_idx: usize = 0;
        for (0..batch_rank) |d| {
            const ac = if (a_batch_shape[d] == 1) 0 else coord[d];
            const bc = if (b_batch_shape[d] == 1) 0 else coord[d];
            a_batch_idx += ac * a_batch_strides[d];
            b_batch_idx += bc * b_batch_strides[d];
        }

        const q_off = a_batch_idx * a_batch_bytes;
        const kt_off = b_batch_idx * b_batch_bytes;
        const out_off = bi * out_batch_bytes;
        const scale_off = bi * scale_batch_bytes;
        const mask_off = bi * mask_batch_bytes;
        switch (device) {
            .cpu => try cpu.attention_scores_offset(allocator, dtype, q, k_t, scale, mask, out, q_off, kt_off, scale_off, mask_off, out_off, m, n, k, mask_fill_value),
            .metal => try metal.attention_scores_offset(dtype, q, k_t, scale, mask, out, q_off, kt_off, scale_off, mask_off, out_off, m, n, k, mask_fill_value),
        }
    }
}

pub fn matmulAdd(
    _: std.mem.Allocator,
    device: Device,
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    bias: *const Storage,
    out: *Storage,
    a_shape: []const usize,
    b_shape: []const usize,
    out_shape: []const usize,
) !void {
    _ = out_shape;
    var aa = a_shape;
    var bb = b_shape;
    if (aa.len == 1) aa = &.{ 1, aa[0] };
    if (bb.len == 1) bb = &.{ bb[0], 1 };
    if (aa.len < 2 or bb.len < 2) return error.ExecutionNotImplemented;
    if (aa[aa.len - 1] != bb[bb.len - 2]) return error.ShapeMismatch;

    const a_batch = aa[0 .. aa.len - 2];
    const b_batch = bb[0 .. bb.len - 2];
    const batch_rank = @max(a_batch.len, b_batch.len);
    if (batch_rank > 8) return error.ExecutionNotImplemented;

    var out_batch_shape: [8]usize = [_]usize{1} ** 8;
    var a_batch_shape: [8]usize = [_]usize{1} ** 8;
    var b_batch_shape: [8]usize = [_]usize{1} ** 8;
    for (0..batch_rank) |i| {
        const ai = if (i + a_batch.len >= batch_rank) a_batch[i + a_batch.len - batch_rank] else 1;
        const bi = if (i + b_batch.len >= batch_rank) b_batch[i + b_batch.len - batch_rank] else 1;
        if (ai != bi and ai != 1 and bi != 1) return error.ExecutionNotImplemented;
        out_batch_shape[i] = @max(ai, bi);
        a_batch_shape[i] = ai;
        b_batch_shape[i] = bi;
    }

    var out_batch_strides: [8]usize = [_]usize{1} ** 8;
    var a_batch_strides: [8]usize = [_]usize{1} ** 8;
    var b_batch_strides: [8]usize = [_]usize{1} ** 8;
    if (batch_rank > 0) {
        out_batch_strides[batch_rank - 1] = 1;
        a_batch_strides[batch_rank - 1] = 1;
        b_batch_strides[batch_rank - 1] = 1;
        var d = batch_rank - 1;
        while (d > 0) {
            d -= 1;
            out_batch_strides[d] = out_batch_strides[d + 1] * out_batch_shape[d + 1];
            a_batch_strides[d] = a_batch_strides[d + 1] * a_batch_shape[d + 1];
            b_batch_strides[d] = b_batch_strides[d + 1] * b_batch_shape[d + 1];
        }
    }

    var batch: usize = 1;
    for (out_batch_shape[0..batch_rank]) |d| batch *= d;
    const m = aa[aa.len - 2];
    const k = aa[aa.len - 1];
    const n = bb[bb.len - 1];
    const elem_bytes: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => @sizeOf(i64),
    };
    const a_batch_bytes = m * k * elem_bytes;
    const b_batch_bytes = k * n * elem_bytes;
    const out_batch_bytes = m * n * elem_bytes;
    const bias_batch_bytes = m * n * elem_bytes;

    var coord: [8]usize = [_]usize{0} ** 8;
    for (0..batch) |bi| {
        var rem = bi;
        for (0..batch_rank) |r| {
            const d = batch_rank - 1 - r;
            coord[d] = rem % out_batch_shape[d];
            rem /= out_batch_shape[d];
        }

        var a_batch_idx: usize = 0;
        var b_batch_idx: usize = 0;
        for (0..batch_rank) |d| {
            const ac = if (a_batch_shape[d] == 1) 0 else coord[d];
            const bc = if (b_batch_shape[d] == 1) 0 else coord[d];
            a_batch_idx += ac * a_batch_strides[d];
            b_batch_idx += bc * b_batch_strides[d];
        }
        const a_off = a_batch_idx * a_batch_bytes;
        const b_off = b_batch_idx * b_batch_bytes;
        const out_off = bi * out_batch_bytes;
        const bias_off = bi * bias_batch_bytes;

        switch (device) {
            .cpu => try cpu.matmul_add_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
            .metal => try metal.matmul_add_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
        }
    }
}

pub fn matmulAddGelu(
    _: std.mem.Allocator,
    device: Device,
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    bias: *const Storage,
    out: *Storage,
    a_shape: []const usize,
    b_shape: []const usize,
    out_shape: []const usize,
) !void {
    if (dtype == .i64) return error.ExecutionNotImplemented;
    return matmulAddLike(
        device,
        dtype,
        a,
        b,
        bias,
        out,
        a_shape,
        b_shape,
        out_shape,
        true,
    );
}

fn matmulAddLike(
    device: Device,
    dtype: DType,
    a: *const Storage,
    b: *const Storage,
    bias: *const Storage,
    out: *Storage,
    a_shape: []const usize,
    b_shape: []const usize,
    out_shape: []const usize,
    with_gelu: bool,
) !void {
    _ = out_shape;
    var aa = a_shape;
    var bb = b_shape;
    if (aa.len == 1) aa = &.{ 1, aa[0] };
    if (bb.len == 1) bb = &.{ bb[0], 1 };
    if (aa.len < 2 or bb.len < 2) return error.ExecutionNotImplemented;
    if (aa[aa.len - 1] != bb[bb.len - 2]) return error.ShapeMismatch;

    const a_batch = aa[0 .. aa.len - 2];
    const b_batch = bb[0 .. bb.len - 2];
    const batch_rank = @max(a_batch.len, b_batch.len);
    if (batch_rank > 8) return error.ExecutionNotImplemented;

    var out_batch_shape: [8]usize = [_]usize{1} ** 8;
    var a_batch_shape: [8]usize = [_]usize{1} ** 8;
    var b_batch_shape: [8]usize = [_]usize{1} ** 8;
    for (0..batch_rank) |i| {
        const ai = if (i + a_batch.len >= batch_rank) a_batch[i + a_batch.len - batch_rank] else 1;
        const bi = if (i + b_batch.len >= batch_rank) b_batch[i + b_batch.len - batch_rank] else 1;
        if (ai != bi and ai != 1 and bi != 1) return error.ExecutionNotImplemented;
        out_batch_shape[i] = @max(ai, bi);
        a_batch_shape[i] = ai;
        b_batch_shape[i] = bi;
    }

    var out_batch_strides: [8]usize = [_]usize{1} ** 8;
    var a_batch_strides: [8]usize = [_]usize{1} ** 8;
    var b_batch_strides: [8]usize = [_]usize{1} ** 8;
    if (batch_rank > 0) {
        out_batch_strides[batch_rank - 1] = 1;
        a_batch_strides[batch_rank - 1] = 1;
        b_batch_strides[batch_rank - 1] = 1;
        var d = batch_rank - 1;
        while (d > 0) {
            d -= 1;
            out_batch_strides[d] = out_batch_strides[d + 1] * out_batch_shape[d + 1];
            a_batch_strides[d] = a_batch_strides[d + 1] * a_batch_shape[d + 1];
            b_batch_strides[d] = b_batch_strides[d + 1] * b_batch_shape[d + 1];
        }
    }

    var batch: usize = 1;
    for (out_batch_shape[0..batch_rank]) |d| batch *= d;
    const m = aa[aa.len - 2];
    const k = aa[aa.len - 1];
    const n = bb[bb.len - 1];
    const elem_bytes: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => @sizeOf(i64),
    };
    const a_batch_bytes = m * k * elem_bytes;
    const b_batch_bytes = k * n * elem_bytes;
    const out_batch_bytes = m * n * elem_bytes;
    const bias_batch_bytes = m * n * elem_bytes;

    var coord: [8]usize = [_]usize{0} ** 8;
    for (0..batch) |bi| {
        var rem = bi;
        for (0..batch_rank) |r| {
            const d = batch_rank - 1 - r;
            coord[d] = rem % out_batch_shape[d];
            rem /= out_batch_shape[d];
        }

        var a_batch_idx: usize = 0;
        var b_batch_idx: usize = 0;
        for (0..batch_rank) |d| {
            const ac = if (a_batch_shape[d] == 1) 0 else coord[d];
            const bc = if (b_batch_shape[d] == 1) 0 else coord[d];
            a_batch_idx += ac * a_batch_strides[d];
            b_batch_idx += bc * b_batch_strides[d];
        }
        const a_off = a_batch_idx * a_batch_bytes;
        const b_off = b_batch_idx * b_batch_bytes;
        const out_off = bi * out_batch_bytes;
        const bias_off = bi * bias_batch_bytes;
        if (with_gelu) {
            switch (device) {
                .cpu => try cpu.matmul_add_gelu_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
                .metal => try metal.matmul_add_gelu_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
            }
        } else {
            switch (device) {
                .cpu => try cpu.matmul_add_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
                .metal => try metal.matmul_add_offset(dtype, a, b, bias, out, a_off, b_off, bias_off, out_off, m, n, k),
            }
        }
    }
}

pub fn unaryChain(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    input: *const Storage,
    out: *Storage,
    shape: []const usize,
    stages: []const UnaryStage,
) !void {
    if (stages.len == 0) return error.InvalidInputCount;
    if (device == .cpu) {
        var cpu_stages = try allocator.alloc(@import("cpu/chain.zig").UnaryStage, stages.len);
        defer allocator.free(cpu_stages);
        for (stages, 0..) |s, i| {
            cpu_stages[i] = .{
                .tag = s.tag,
                .clamp_min = s.clamp_min,
                .clamp_max = s.clamp_max,
            };
        }
        return cpu.unary_chain(dtype, input, out, cpu_stages);
    }
    if (device == .metal) {
        var metal_stages = try allocator.alloc(@import("metal/chain.zig").UnaryStage, stages.len);
        defer allocator.free(metal_stages);
        for (stages, 0..) |s, i| {
            metal_stages[i] = .{
                .tag = s.tag,
                .clamp_min = s.clamp_min,
                .clamp_max = s.clamp_max,
            };
        }
        return metal.unary_chain(dtype, input, out, metal_stages);
    }
    const temp_bytes = try shapeBytes(dtype, shape);

    var current: *const Storage = input;
    var i: usize = 0;
    while (i < stages.len) : (i += 1) {
        const is_last = i + 1 == stages.len;
        const next = if (is_last) out else try createTempStorage(allocator, device, temp_bytes);
        if (!is_last) {
            defer next.release();
        }

        const s = stages[i];
        if (s.tag == .clamp) {
            const min_v = s.clamp_min orelse return error.InvalidClampBounds;
            const max_v = s.clamp_max orelse return error.InvalidClampBounds;
            try unaryClamp(device, dtype, current, next, min_v, max_v);
        } else {
            try unary(device, s.tag, dtype, current, next);
        }
        current = next;
    }
}

pub fn binaryThenUnaryChain(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    binary_tag: OpTag,
    lhs: *const Storage,
    rhs: *const Storage,
    out: *Storage,
    shape: []const usize,
    stages: []const UnaryStage,
) !void {
    if (stages.len == 0) return error.InvalidInputCount;
    if (device == .cpu) {
        var cpu_stages = try allocator.alloc(@import("cpu/chain.zig").UnaryStage, stages.len);
        defer allocator.free(cpu_stages);
        for (stages, 0..) |s, i| {
            cpu_stages[i] = .{
                .tag = s.tag,
                .clamp_min = s.clamp_min,
                .clamp_max = s.clamp_max,
            };
        }
        return cpu.binary_then_unary_chain(dtype, binary_tag, lhs, rhs, out, cpu_stages);
    }
    if (device == .metal) {
        var metal_stages = try allocator.alloc(@import("metal/chain.zig").UnaryStage, stages.len);
        defer allocator.free(metal_stages);
        for (stages, 0..) |s, i| {
            metal_stages[i] = .{
                .tag = s.tag,
                .clamp_min = s.clamp_min,
                .clamp_max = s.clamp_max,
            };
        }
        return metal.binary_then_unary_chain(dtype, binary_tag, lhs, rhs, out, metal_stages);
    }
    const temp_bytes = try shapeBytes(dtype, shape);
    var first = try createTempStorage(allocator, device, temp_bytes);
    defer first.release();
    try binary(device, binary_tag, dtype, lhs, rhs, first);
    try unaryChain(allocator, device, dtype, first, out, shape, stages);
}

fn shapeBytes(dtype: DType, shape: []const usize) !usize {
    var elems: usize = 1;
    for (shape) |d| elems = try std.math.mul(usize, elems, d);
    const elem_bytes: usize = switch (dtype) {
        .f32 => @sizeOf(f32),
        .f64 => @sizeOf(f64),
        .i64 => @sizeOf(i64),
    };
    return std.math.mul(usize, elems, elem_bytes);
}

fn createTempStorage(allocator: std.mem.Allocator, device: Device, bytes: usize) !*Storage {
    return switch (device) {
        .cpu => Storage.createCpuWithMetadata(allocator, bytes, false, .{
            .policy = .scratch,
            .reason = .workspace,
            .source = .graph,
        }),
        .metal => Storage.createMetalWithMetadata(allocator, bytes, .{
            .policy = .scratch,
            .reason = .workspace,
            .source = .graph,
        }),
    };
}
pub fn layerNorm(device: Device, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
    switch (device) {
        .cpu => try cpu.layer_norm(dtype, input, out, shape, axis, eps),
        .metal => try metal.layer_norm(dtype, input, out, shape, axis, eps),
    }
}
pub fn rmsNorm(device: Device, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, axis: usize, eps: f64) !void {
    switch (device) {
        .cpu => try cpu.rms_norm(dtype, input, out, shape, axis, eps),
        .metal => try metal.rms_norm(dtype, input, out, shape, axis, eps),
    }
}
pub fn reductionAxis(
    device: Device,
    tag: OpTag,
    input_dtype: DType,
    input: *const Storage,
    out: *Storage,
    shape: []const usize,
    axis: usize,
    keepdim: bool,
) !void {
    if (axis >= shape.len) return error.InvalidAxis;
    var out_elems: usize = 1;
    if (shape.len > 0) {
        for (shape, 0..) |dim, i| {
            if (!keepdim and i == axis) continue;
            if (keepdim and i == axis) {
                out_elems *= 1;
                continue;
            }
            out_elems *= dim;
        }
    }

    switch (device) {
        .cpu => try cpu.reduction_axis(tag, input_dtype, input, out, shape, axis, keepdim),
        .metal => {
            if (shape.len == 2) {
                try metal.reduction_axis(tag, input_dtype, input, out, shape[0], shape[1], axis, out_elems);
            } else if (shape.len == 1) {
                // Lift rank-1 vector into a synthetic 1xN matrix, reducing across axis=1.
                try metal.reduction_axis(tag, input_dtype, input, out, 1, shape[0], 1, out_elems);
            } else {
                try @import("metal/reduction_axis.zig").runNd(tag, input_dtype, input, out, shape, axis, out_elems);
            }
        },
    }
}

pub fn contiguous(device: Device, dtype: DType, input: *const Storage, out: *Storage, shape: []const usize, strides: []const isize, offset: usize) !void {
    switch (device) {
        .cpu => try cpu.contiguous(dtype, input, out, shape, strides, offset),
        .metal => try metal.contiguous(dtype, input, out, shape, strides, offset),
    }
}

pub fn dot(device: Device, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage) !void {
    switch (device) {
        .cpu => try cpu.dot(dtype, a, b, out),
        .metal => try metal.dot(dtype, a, b, out),
    }
}

pub fn matmul(device: Device, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, a_shape: []const usize, b_shape: []const usize) !void {
    var a_shape_obj = try @import("../types/tensor/shape.zig").Shape.initCopy(std.heap.page_allocator, a_shape);
    defer a_shape_obj.deinit();
    var b_shape_obj = try @import("../types/tensor/shape.zig").Shape.initCopy(std.heap.page_allocator, b_shape);
    defer b_shape_obj.deinit();
    var a_layout = try Layout.initContiguous(std.heap.page_allocator, a_shape_obj);
    defer a_layout.deinit();
    var b_layout = try Layout.initContiguous(std.heap.page_allocator, b_shape_obj);
    defer b_layout.deinit();
    return matmulWithLayouts(device, dtype, a, b, out, a_shape, a_layout, b_shape, b_layout);
}

pub fn matmulWithLayouts(device: Device, dtype: DType, a: *const Storage, b: *const Storage, out: *Storage, a_shape: []const usize, a_layout: Layout, b_shape: []const usize, b_layout: Layout) !void {
    const lhs = try normalizeMatmulOperand(a_shape, a_layout, .lhs);
    const rhs = try normalizeMatmulOperand(b_shape, b_layout, .rhs);
    if (lhs.k != rhs.shape[rhs.rank - 2]) return error.ShapeMismatch;
    const m = lhs.m;
    const k = lhs.k;
    const n = rhs.n;

    const batch_rank = @max(lhs.batch_rank, rhs.batch_rank);
    if (batch_rank > 8) return error.ExecutionNotImplemented;

    var out_batch_shape: [8]usize = [_]usize{1} ** 8;
    for (0..batch_rank) |i| {
        const ai = if (i + lhs.batch_rank >= batch_rank) lhs.shape[i + lhs.batch_rank - batch_rank] else 1;
        const bi = if (i + rhs.batch_rank >= batch_rank) rhs.shape[i + rhs.batch_rank - batch_rank] else 1;
        if (ai != bi and ai != 1 and bi != 1) return error.ExecutionNotImplemented;
        out_batch_shape[i] = @max(ai, bi);
    }

    var batch: usize = 1;
    for (out_batch_shape[0..batch_rank]) |d| batch *= d;
    const elem_bytes = dtype.size();
    const out_batch_bytes = m * n * elem_bytes;

    if (device == .metal and batch > 1) {
        const a_offsets_bytes = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(a_offsets_bytes);
        const b_offsets_bytes = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(b_offsets_bytes);
        const out_offsets_bytes = try std.heap.page_allocator.alloc(usize, batch);
        defer std.heap.page_allocator.free(out_offsets_bytes);

        var coord_many: [8]usize = [_]usize{0} ** 8;
        for (0..batch) |bi| {
            var rem = bi;
            for (0..batch_rank) |r| {
                const d = batch_rank - 1 - r;
                coord_many[d] = rem % out_batch_shape[d];
                rem /= out_batch_shape[d];
            }

            const a_batch_offset = lhs.offset + computeBroadcastBatchOffset(batch_rank, coord_many[0..batch_rank], lhs.batch_rank, lhs.shape[0..lhs.batch_rank], lhs.strides[0..lhs.batch_rank]);
            const b_batch_offset = rhs.offset + computeBroadcastBatchOffset(batch_rank, coord_many[0..batch_rank], rhs.batch_rank, rhs.shape[0..rhs.batch_rank], rhs.strides[0..rhs.batch_rank]);
            a_offsets_bytes[bi] = a_batch_offset * elem_bytes;
            b_offsets_bytes[bi] = b_batch_offset * elem_bytes;
            out_offsets_bytes[bi] = bi * out_batch_bytes;
        }

        try metal.matmul_strided_many(
            dtype,
            a,
            b,
            out,
            a_offsets_bytes,
            b_offsets_bytes,
            out_offsets_bytes,
            lhs.row_stride,
            lhs.col_stride,
            rhs.row_stride,
            rhs.col_stride,
            m,
            n,
            k,
        );
        return;
    }

    var coord: [8]usize = [_]usize{0} ** 8;
    for (0..batch) |bi| {
        var rem = bi;
        for (0..batch_rank) |r| {
            const d = batch_rank - 1 - r;
            coord[d] = rem % out_batch_shape[d];
            rem /= out_batch_shape[d];
        }

        const a_batch_offset = lhs.offset + computeBroadcastBatchOffset(batch_rank, coord[0..batch_rank], lhs.batch_rank, lhs.shape[0..lhs.batch_rank], lhs.strides[0..lhs.batch_rank]);
        const b_batch_offset = rhs.offset + computeBroadcastBatchOffset(batch_rank, coord[0..batch_rank], rhs.batch_rank, rhs.shape[0..rhs.batch_rank], rhs.strides[0..rhs.batch_rank]);
        const out_offset_bytes = bi * out_batch_bytes;
        const a_offset_bytes = a_batch_offset * elem_bytes;
        const b_offset_bytes = b_batch_offset * elem_bytes;

        switch (device) {
            .cpu => try cpu.matmul_strided(
                dtype,
                a,
                b,
                out,
                a_offset_bytes,
                lhs.row_stride,
                lhs.col_stride,
                b_offset_bytes,
                rhs.row_stride,
                rhs.col_stride,
                out_offset_bytes,
                m,
                n,
                k,
            ),
            .metal => try metal.matmul_strided(
                dtype,
                a,
                b,
                out,
                a_offset_bytes,
                b_offset_bytes,
                out_offset_bytes,
                lhs.row_stride,
                lhs.col_stride,
                rhs.row_stride,
                rhs.col_stride,
                m,
                n,
                k,
            ),
        }
    }
}

pub fn gather(
    device: Device,
    dtype: DType,
    input: *const Storage,
    index: *const Storage,
    out: *Storage,
    input_shape: []const usize,
    out_shape: []const usize,
    axis: usize,
) !void {
    switch (device) {
        .cpu => try cpu.gather(dtype, input, index, out, input_shape, axis),
        .metal => try metal.gather(dtype, input, index, out, out_shape, axis, input_shape[axis]),
    }
}

pub fn embedding(device: Device, dtype: DType, table: *const Storage, index: *const Storage, out: *Storage, table_shape: []const usize, index_shape: []const usize) !void {
    switch (device) {
        .cpu => try cpu.embedding(dtype, table, index, out, table_shape, index_shape),
        .metal => try metal.embedding(dtype, table, index, out, table_shape, index_shape),
    }
}

pub fn indexSelect(device: Device, dtype: DType, input: *const Storage, index: *const Storage, out: *Storage, shape: []const usize, axis: usize, index_len: usize) !void {
    switch (device) {
        .cpu => try cpu.index_select(dtype, input, index, out, shape, axis, index_len),
        .metal => try metal.index_select(dtype, input, index, out, shape, axis, index_len),
    }
}

pub fn scatterAdd(device: Device, dtype: DType, index_dtype: DType, base: *const Storage, index: *const Storage, updates: *const Storage, out: *Storage, shape: []const usize, axis: usize) !void {
    switch (device) {
        .cpu => try cpu.scatter_add(dtype, base, index, updates, out, shape, axis),
        .metal => try metal.scatter_add(dtype, index_dtype, base, index, updates, out, shape, axis),
    }
}

pub fn oneHot(device: Device, index: *const Storage, out: *Storage, num_classes: usize) !void {
    switch (device) {
        .cpu => try cpu.one_hot(index, out, num_classes),
        .metal => try metal.one_hot(index, out, num_classes),
    }
}

pub fn topk(
    allocator: std.mem.Allocator,
    device: Device,
    dtype: DType,
    input: *const Storage,
    values_out: *Storage,
    indices_out: *Storage,
    shape: []const usize,
    axis: usize,
    k: usize,
    largest: bool,
) !void {
    switch (device) {
        .cpu => try cpu.topk(allocator, dtype, input, values_out, indices_out, shape, axis, k, largest),
        .metal => try metal.topk(dtype, input, values_out, indices_out, shape, axis, k, largest),
    }
}

pub fn cat(device: Device, dtype: DType, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    switch (device) {
        .cpu => try cpu.cat(dtype, inputs, out, in_shape, axis),
        .metal => try metal.cat(dtype, inputs, out, in_shape, axis),
    }
}

pub fn stack(device: Device, dtype: DType, inputs: []const *const Storage, out: *Storage, in_shape: []const usize, axis: usize) !void {
    switch (device) {
        .cpu => try cpu.stack(dtype, inputs, out, in_shape, axis),
        .metal => try metal.stack(dtype, inputs, out, in_shape, axis),
    }
}

pub fn slice(device: Device, dtype: DType, input: *const Storage, out: *Storage, in_shape: []const usize, ranges: []const @import("../types/operation/options.zig").SliceRange) !void {
    switch (device) {
        .cpu => try cpu.slice(dtype, input, out, in_shape, ranges),
        .metal => try metal.slice(dtype, input, out, in_shape, ranges),
    }
}

test "metal backend is routed through kernel dispatch surface" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    const a_handle = try a.metalHandle();
    const b_handle = try b.metalHandle();
    const out_handle = try out.metalHandle();

    const av = [_]f32{2.0};
    const bv = [_]f32{3.5};
    try metal.common.writeBuffer(a_handle, @import("std").mem.asBytes(&av));
    try metal.common.writeBuffer(b_handle, @import("std").mem.asBytes(&bv));

    try binary(.metal, .add, .f32, a, b, out);

    var ov = [_]f32{0.0};
    try metal.common.readBuffer(out_handle, @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 5.5), ov[0], 1e-5);
}

test "metal binary supports i64 native path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 7, 8, -9, 10 }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 2, -3, 4, 5 }));

    try binary(.metal, .mul, .i64, a, b, out);
    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 14, -24, -36, 50 }, &ov);
}

test "metal unary and clamp support i64 native path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const tmp = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer tmp.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ -4, 3, -2, 9 }));

    try unary(.metal, .abs, .i64, input, tmp);
    try unaryClamp(.metal, .i64, tmp, out, 0, 3);

    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 3, 3, 2, 3 }, &ov);
}

test "cpu clamp supports i64 with integral bounds" {
    const allocator = @import("std").testing.allocator;
    const input = try Storage.createCpuWithMetadata(allocator, 4 * @sizeOf(i64), false, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createCpuWithMetadata(allocator, 4 * @sizeOf(i64), false, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    const inb = try input.writableBytes();
    const inv = @import("std").mem.bytesAsSlice(i64, inb);
    @memcpy(inv, &[_]i64{ -5, -1, 3, 10 });

    try unaryClamp(.cpu, .i64, input, out, -2, 4);
    const outb = try out.readableBytes();
    const ov = @import("std").mem.bytesAsSlice(i64, outb);
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ -2, -1, 3, 4 }, ov);
}

test "cpu clamp rejects non-integral bounds for i64" {
    const allocator = @import("std").testing.allocator;
    const input = try Storage.createCpuWithMetadata(allocator, 2 * @sizeOf(i64), false, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createCpuWithMetadata(allocator, 2 * @sizeOf(i64), false, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try @import("std").testing.expectError(error.InvalidClampBounds, unaryClamp(.cpu, .i64, input, out, -1.5, 2));
}

test "metal where supports i64 condition tensor" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const cond = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer cond.release();
    const on_true = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer on_true.release();
    const on_false = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer on_false.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try cond.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 0, 1, 0, 1 }));
    try metal.common.writeBuffer(try on_true.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 10, 20, 30, 40 }));
    try metal.common.writeBuffer(try on_false.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));

    try where(.metal, .i64, .f32, cond, on_true, on_false, out);
    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 20), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 3), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 40), ov[3]);
}

test "metal where supports i64 values" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const cond = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer cond.release();
    const on_true = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer on_true.release();
    const on_false = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer on_false.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try cond.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 0, 0, 1 }));
    try metal.common.writeBuffer(try on_true.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 10, 20, 30, 40 }));
    try metal.common.writeBuffer(try on_false.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4 }));
    try where(.metal, .i64, .i64, cond, on_true, on_false, out);

    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 10), ov[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(i64, 3), ov[2]);
    try @import("std").testing.expectEqual(@as(i64, 40), ov[3]);
}

test "metal reductionAll mean and min are routed through kernel dispatch surface" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    const in_handle = try in.metalHandle();
    const out_handle = try out.metalHandle();
    const iv = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    try metal.common.writeBuffer(in_handle, @import("std").mem.asBytes(&iv));

    try reductionAll(.metal, .mean_all, .f32, in, out);
    var ov = [_]f32{0.0};
    try metal.common.readBuffer(out_handle, @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 5.0), ov[0], 1e-5);

    try reductionAll(.metal, .min_all, .f32, in, out);
    try metal.common.readBuffer(out_handle, @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2.0), ov[0], 1e-5);

    try reductionAll(.metal, .variance_all, .f32, in, out);
    try metal.common.readBuffer(out_handle, @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 5.0), ov[0], 1e-5);

    try reductionAll(.metal, .std_all, .f32, in, out);
    try metal.common.readBuffer(out_handle, @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2.2360679), ov[0], 1e-4);
}

test "metal reductionAll argmin/argmax produce native i64 outputs" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 5 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 3, -5, 2, 9, 1 }));
    try reductionAll(.metal, .argmin_all, .f32, in, out);
    var idx = [_]i64{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);

    try reductionAll(.metal, .argmax_all, .f32, in, out);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 3), idx[0]);
}

test "metal reductionAll supports i64 sum/min/max natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 5 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 3, -5, 2, 9, 1 }));
    try reductionAll(.metal, .sum_all, .i64, in, out);
    var ov = [_]i64{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 10), ov[0]);

    try reductionAll(.metal, .min_all, .i64, in, out);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, -5), ov[0]);

    try reductionAll(.metal, .max_all, .i64, in, out);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 9), ov[0]);
}

test "metal reductionAll supports i64 argmin/argmax natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 5 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 3, -5, 2, 9, 1 }));
    try reductionAll(.metal, .argmin_all, .i64, in, out);
    var idx = [_]i64{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);

    try reductionAll(.metal, .argmax_all, .i64, in, out);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 3), idx[0]);
}

test "metal reductionAll rejects unsupported i64 stats ops" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4 }));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAll(.metal, .mean_all, .i64, in, out));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAll(.metal, .variance_all, .i64, in, out));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAll(.metal, .std_all, .i64, in, out));
}

test "metal reductionAxis argmin/argmax produce native i64 outputs for rank-2" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        3, 1, 2,
        0, 9, -7,
    }));
    try reductionAxis(.metal, .argmin_axis, .f32, in, out, &.{ 2, 3 }, 1, false);
    var idx = [_]i64{ 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[1]);

    try reductionAxis(.metal, .argmax_axis, .f32, in, out, &.{ 2, 3 }, 1, false);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 0), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[1]);
}

test "metal reductionAxis variance/std are routed for rank-2" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2, 3,
        4, 5, 6,
    }));
    try reductionAxis(.metal, .variance_axis, .f32, in, out, &.{ 2, 3 }, 1, false);
    var ov = [_]f32{ 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.6666667), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.6666667), ov[1], 1e-5);

    try reductionAxis(.metal, .std_axis, .f32, in, out, &.{ 2, 3 }, 1, false);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.8164966), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.8164966), ov[1], 1e-5);
}

test "metal reductionAxis supports rank-1 vector" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 2, 4, 6, 8 }));

    try reductionAxis(.metal, .sum_axis, .f32, in, out, &.{4}, 0, false);
    var ov = [_]f32{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 20), ov[0], 1e-5);

    try reductionAxis(.metal, .max_axis, .f32, in, out, &.{4}, 0, false);
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), ov[0], 1e-5);
}

test "metal reductionAxis supports i64 sum/min/max for rank-2 and rank-3 nd" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in2 = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in2.release();
    const out2 = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out2.release();

    try metal.common.writeBuffer(try in2.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        3, 1, 2,
        0, 9, -7,
    }));
    try reductionAxis(.metal, .sum_axis, .i64, in2, out2, &.{ 2, 3 }, 1, false);
    var ov2 = [_]i64{ 0, 0 };
    try metal.common.readBuffer(try out2.metalHandle(), @import("std").mem.asBytes(&ov2));
    try @import("std").testing.expectEqual(@as(i64, 6), ov2[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), ov2[1]);

    try reductionAxis(.metal, .min_axis, .i64, in2, out2, &.{ 2, 3 }, 1, false);
    try metal.common.readBuffer(try out2.metalHandle(), @import("std").mem.asBytes(&ov2));
    try @import("std").testing.expectEqual(@as(i64, 1), ov2[0]);
    try @import("std").testing.expectEqual(@as(i64, -7), ov2[1]);

    const in3 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in3.release();
    const out3 = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out3.release();
    try metal.common.writeBuffer(try in3.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    }));
    try reductionAxis(.metal, .max_axis, .i64, in3, out3, &.{ 2, 2, 2 }, 2, false);
    var ov3 = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out3.metalHandle(), @import("std").mem.asBytes(&ov3));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 2, 4, 6, 8 }, &ov3);
}

test "metal reductionAxis supports i64 argmin/argmax for rank-2 and rank-3 nd" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in2 = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in2.release();
    const out2 = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out2.release();

    try metal.common.writeBuffer(try in2.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        3, 1, 2,
        0, 9, -7,
    }));
    try reductionAxis(.metal, .argmin_axis, .i64, in2, out2, &.{ 2, 3 }, 1, false);
    var ov2 = [_]i64{ 0, 0 };
    try metal.common.readBuffer(try out2.metalHandle(), @import("std").mem.asBytes(&ov2));
    try @import("std").testing.expectEqual(@as(i64, 1), ov2[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), ov2[1]);

    const in3 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in3.release();
    const out3 = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out3.release();
    try metal.common.writeBuffer(try in3.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    }));
    try reductionAxis(.metal, .argmax_axis, .i64, in3, out3, &.{ 2, 2, 2 }, 2, false);
    var ov3 = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out3.metalHandle(), @import("std").mem.asBytes(&ov3));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 1, 1, 1, 1 }, &ov3);
}

test "metal reductionAxis rejects unsupported i64 stats ops" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 2, 3,
        4, 5, 6,
    }));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAxis(.metal, .mean_axis, .i64, in, out, &.{ 2, 3 }, 1, false));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAxis(.metal, .variance_axis, .i64, in, out, &.{ 2, 3 }, 1, false));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAxis(.metal, .std_axis, .i64, in, out, &.{ 2, 3 }, 1, false));
}

test "metal softmax rejects i64 dtype explicitly" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();
    var input_shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ 2, 2 });
    defer input_shape.deinit();
    var input_layout = try Layout.initContiguous(allocator, input_shape);
    defer input_layout.deinit();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4 }));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, softmax(.metal, .i64, input, out, &.{ 2, 2 }, input_layout, 1));
}

test "metal f64 remains explicit not-implemented on current backend" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out_vec = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out_vec.release();
    const out_scalar = try Storage.createMetalWithMetadata(allocator, @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out_scalar.release();
    var dense_shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ 2, 2 });
    defer dense_shape.deinit();
    var dense_layout = try Layout.initContiguous(allocator, dense_shape);
    defer dense_layout.deinit();

    try @import("std").testing.expectError(error.ExecutionNotImplemented, binary(.metal, .add, .f64, a, b, out_vec));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, dot(.metal, .f64, a, b, out_scalar));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, matmul(.metal, .f64, a, b, out_vec, &.{ 2, 2 }, &.{ 2, 2 }));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, reductionAll(.metal, .sum_all, .f64, a, out_scalar));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, softmax(.metal, .f64, a, out_vec, &.{ 2, 2 }, dense_layout, 1));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, layerNorm(.metal, .f64, a, out_vec, &.{ 2, 2 }, 1, 1e-5));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, unary(.metal, .abs, .f64, a, out_vec));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, unaryClamp(.metal, .f64, a, out_vec, -1.0, 1.0));

    // where
    const cond = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer cond.release();
    try @import("std").testing.expectError(error.ExecutionNotImplemented, where(.metal, .i64, .f64, cond, a, b, out_vec));

    // gather/index_select
    const index = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out_two = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out_two.release();
    try @import("std").testing.expectError(error.ExecutionNotImplemented, gather(.metal, .f64, a, index, out_two, &.{ 2, 2 }, &.{ 2, 2 }, 1));
    try @import("std").testing.expectError(error.ExecutionNotImplemented, indexSelect(.metal, .f64, a, index, out_two, &.{ 2, 2 }, 0, 2));

    // topk
    const values = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();
    try @import("std").testing.expectError(error.ExecutionNotImplemented, topk(@import("std").testing.allocator, .metal, .f64, a, values, indices, &.{ 2, 2 }, 1, 1, true));

    // slice
    const ranges = [_]@import("../types/operation/options.zig").SliceRange{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 0, .stop = 1, .step = 1 },
    };
    try @import("std").testing.expectError(error.ExecutionNotImplemented, slice(.metal, .f64, a, out_two, &.{ 2, 2 }, &ranges));
}

test "metal reductionAxis supports rank-3 via native nd path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const in = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer in.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();
    const out_idx = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out_idx.release();

    try metal.common.writeBuffer(try in.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    }));
    try reductionAxis(.metal, .sum_axis, .f32, in, out, &.{ 2, 2, 2 }, 2, false);
    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 11), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 15), ov[3], 1e-5);

    try reductionAxis(.metal, .argmax_axis, .f32, in, out_idx, &.{ 2, 2, 2 }, 2, false);
    var idx = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out_idx.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[1]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[2]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[3]);
}

test "metal softmax and matmul are routed through kernel dispatch surface" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;

    const sm_in = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer sm_in.release();
    const sm_out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer sm_out.release();

    const sm_in_h = try sm_in.metalHandle();
    const sm_out_h = try sm_out.metalHandle();
    const sm_vals = [_]f32{ 1, 2, 3, 4 };
    try metal.common.writeBuffer(sm_in_h, @import("std").mem.asBytes(&sm_vals));
    var sm_shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ 2, 2 });
    defer sm_shape.deinit();
    var sm_layout = try Layout.initContiguous(allocator, sm_shape);
    defer sm_layout.deinit();
    try softmax(.metal, .f32, sm_in, sm_out, &.{ 2, 2 }, sm_layout, 1);
    var sm_read = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(sm_out_h, @import("std").mem.asBytes(&sm_read));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.26894143), sm_read[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.7310586), sm_read[1], 1e-5);

    const mm_a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mm_a.release();
    const mm_b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mm_b.release();
    const mm_out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mm_out.release();
    try metal.common.writeBuffer(try mm_a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try mm_b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 5, 6, 7, 8 }));
    try matmul(.metal, .f32, mm_a, mm_b, mm_out, &.{ 2, 2 }, &.{ 2, 2 });
    var mm_read = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try mm_out.metalHandle(), @import("std").mem.asBytes(&mm_read));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 19), mm_read[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 22), mm_read[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 43), mm_read[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 50), mm_read[3], 1e-5);
}

test "metal softmax accepts positive-stride view layout through dispatch" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    var shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try Layout.initCopy(allocator, &.{ 1, 2 }, 0);
    defer layout.deinit();

    try softmax(.metal, .f32, input, out, shape.dims, layout, 1);

    var ov = [_]f32{0} ** 4;
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.11920292), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.88079703), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.11920292), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.88079703), ov[3], 1e-5);
}

test "metal layer_norm is routed through kernel dispatch surface for f32" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 3, 2, 4 }));
    try layerNorm(.metal, .f32, input, out, &.{ 2, 2 }, 1, 1e-5);

    var ov = [_]f32{0} ** 4;
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, -0.999995), ov[0], 1e-4);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.999995), ov[1], 1e-4);
    try @import("std").testing.expectApproxEqAbs(@as(f32, -0.999995), ov[2], 1e-4);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.999995), ov[3], 1e-4);
}

test "metal matmul supports rank-1 variants" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;

    // 1D @ 1D -> scalar
    const d_a = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer d_a.release();
    const d_b = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer d_b.release();
    const d_out = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer d_out.release();
    try metal.common.writeBuffer(try d_a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3 }));
    try metal.common.writeBuffer(try d_b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 4, 5, 6 }));
    try matmul(.metal, .f32, d_a, d_b, d_out, &.{3}, &.{3});
    var scalar = [_]f32{0};
    try metal.common.readBuffer(try d_out.metalHandle(), @import("std").mem.asBytes(&scalar));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 32), scalar[0], 1e-5);

    // 2D @ 1D -> 1D
    const mv_a = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mv_a.release();
    const mv_b = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mv_b.release();
    const mv_out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mv_out.release();
    try metal.common.writeBuffer(try mv_a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4, 5, 6 }));
    try metal.common.writeBuffer(try mv_b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 7, 8, 9 }));
    try matmul(.metal, .f32, mv_a, mv_b, mv_out, &.{ 2, 3 }, &.{3});
    var mv = [_]f32{ 0, 0 };
    try metal.common.readBuffer(try mv_out.metalHandle(), @import("std").mem.asBytes(&mv));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 50), mv[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 122), mv[1], 1e-5);

    // 1D @ 2D -> 1D
    const vm_a = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer vm_a.release();
    const vm_b = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer vm_b.release();
    const vm_out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer vm_out.release();
    try metal.common.writeBuffer(try vm_a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3 }));
    try metal.common.writeBuffer(try vm_b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 4, 5, 6, 7, 8, 9 }));
    try matmul(.metal, .f32, vm_a, vm_b, vm_out, &.{3}, &.{ 3, 2 });
    var vm = [_]f32{ 0, 0 };
    try metal.common.readBuffer(try vm_out.metalHandle(), @import("std").mem.asBytes(&vm));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 40), vm[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 46), vm[1], 1e-5);
}

test "metal matmul supports rank-3 batched path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    // batch 0: [[1,2],[3,4]] @ [[1,0],[0,1]] -> [[1,2],[3,4]]
    // batch 1: [[5,6],[7,8]] @ [[2,0],[0,2]] -> [[10,12],[14,16]]
    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }));
    try matmul(.metal, .f32, a, b, out, &.{ 2, 2, 2 }, &.{ 2, 2, 2 });

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[3], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 10), ov[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 12), ov[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 14), ov[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 16), ov[7], 1e-5);
}

test "metal matmul supports i64" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 2,
        3, 4,
    }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        5, 6,
        7, 8,
    }));
    try matmul(.metal, .i64, a, b, out, &.{ 2, 2 }, &.{ 2, 2 });

    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 19), ov[0]);
    try @import("std").testing.expectEqual(@as(i64, 22), ov[1]);
    try @import("std").testing.expectEqual(@as(i64, 43), ov[2]);
    try @import("std").testing.expectEqual(@as(i64, 50), ov[3]);
}

test "metal matmul supports i64 rank-3 batched path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    // batch 0: [[1,2],[3,4]] @ I -> [[1,2],[3,4]]
    // batch 1: [[5,6],[7,8]] @ 2I -> [[10,12],[14,16]]
    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]i64{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }));
    try matmul(.metal, .i64, a, b, out, &.{ 2, 2, 2 }, &.{ 2, 2, 2 });

    var ov = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(i64, 3), ov[2]);
    try @import("std").testing.expectEqual(@as(i64, 4), ov[3]);
    try @import("std").testing.expectEqual(@as(i64, 10), ov[4]);
    try @import("std").testing.expectEqual(@as(i64, 12), ov[5]);
    try @import("std").testing.expectEqual(@as(i64, 14), ov[6]);
    try @import("std").testing.expectEqual(@as(i64, 16), ov[7]);
}

test "metal matmul supports rank-4 batched path with matching batch dims" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 16 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 16 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 16 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    // 4 batches total (2x2), each is 2x2 @ 2x2.
    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 0, 0, 1,
        2, 0, 0, 2,
        3, 0, 0, 3,
        4, 0, 0, 4,
    }));
    try matmul(.metal, .f32, a, b, out, &.{ 2, 2, 2, 2 }, &.{ 2, 2, 2, 2 });

    var ov = [_]f32{0} ** 16;
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));

    // Batch0 * I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[3], 1e-5);
    // Batch1 * 2I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 10), ov[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 12), ov[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 14), ov[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 16), ov[7], 1e-5);
    // Batch2 * 3I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 27), ov[8], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 30), ov[9], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 33), ov[10], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 36), ov[11], 1e-5);
    // Batch3 * 4I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 52), ov[12], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 56), ov[13], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 60), ov[14], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 64), ov[15], 1e-5);
}

test "metal matmul supports broadcasted batch dims" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    // A shape [1,2,2], B shape [2,2,2] => out [2,2,2], A broadcast across batch.
    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2,
        3, 4,
    }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }));
    try matmul(.metal, .f32, a, b, out, &.{ 1, 2, 2 }, &.{ 2, 2, 2 });

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    // batch0: A * I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[3], 1e-5);
    // batch1: A * 2I
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), ov[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), ov[7], 1e-5);
}

test "metal matmul supports mixed-rank broadcast batches" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;

    // 2D @ 3D : [2,2] @ [2,2,2] => [2,2,2]
    const a2 = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a2.release();
    const b3 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b3.release();
    const out23 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out23.release();

    try metal.common.writeBuffer(try a2.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2,
        3, 4,
    }));
    try metal.common.writeBuffer(try b3.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }));
    try matmul(.metal, .f32, a2, b3, out23, &.{ 2, 2 }, &.{ 2, 2, 2 });

    var ov23 = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out23.metalHandle(), @import("std").mem.asBytes(&ov23));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov23[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov23[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov23[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov23[3], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov23[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov23[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), ov23[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), ov23[7], 1e-5);

    // 3D @ 2D : [2,2,2] @ [2,2] => [2,2,2]
    const a3 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a3.release();
    const b2 = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b2.release();
    const out32 = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out32.release();

    try metal.common.writeBuffer(try a3.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }));
    try metal.common.writeBuffer(try b2.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 0,
        0, 1,
    }));
    try matmul(.metal, .f32, a3, b2, out32, &.{ 2, 2, 2 }, &.{ 2, 2 });

    var ov32 = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out32.metalHandle(), @import("std").mem.asBytes(&ov32));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov32[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov32[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov32[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov32[3], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 5), ov32[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), ov32[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), ov32[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), ov32[7], 1e-5);
}

test "cpu matmul supports broadcasted batch dims" {
    const allocator = @import("std").testing.allocator;
    const a = try Storage.createCpuWithMetadata(allocator, 4 * @sizeOf(f32), true, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createCpuWithMetadata(allocator, 8 * @sizeOf(f32), true, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createCpuWithMetadata(allocator, 8 * @sizeOf(f32), true, .{
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    @memcpy(try a.writableBytes(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    @memcpy(try b.writableBytes(), @import("std").mem.asBytes(&[_]f32{
        1, 0, 0, 1,
        2, 0, 0, 2,
    }));
    try matmul(.cpu, .f32, a, b, out, &.{ 1, 2, 2 }, &.{ 2, 2, 2 });

    const ov = @import("std").mem.bytesAsSlice(f32, try out.readableBytes());
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[3], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), ov[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), ov[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), ov[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), ov[7], 1e-5);
}

test "metal softmax supports rank-1 vector" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3 }));
    var vec_shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{3});
    defer vec_shape.deinit();
    var vec_layout = try Layout.initContiguous(allocator, vec_shape);
    defer vec_layout.deinit();
    try softmax(.metal, .f32, input, out, &.{3}, vec_layout, 0);

    var ov = [_]f32{ 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.09003057), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.24472848), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.66524094), ov[2], 1e-5);
}

test "metal softmax supports rank-3 axis" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    }));
    var nd_shape = try @import("../types/tensor/shape.zig").Shape.initCopy(allocator, &.{ 2, 1, 4 });
    defer nd_shape.deinit();
    var nd_layout = try Layout.initContiguous(allocator, nd_shape);
    defer nd_layout.deinit();
    try softmax(.metal, .f32, input, out, &.{ 2, 1, 4 }, nd_layout, 2);

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.0320586), ov[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.0871443), ov[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.2368828), ov[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.6439142), ov[3], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.0320586), ov[4], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.0871443), ov[5], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.2368828), ov[6], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 0.6439142), ov[7], 1e-5);
}

test "metal topk returns values and i64 indices natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const values = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 9, 3, 10, 7, 6 }));
    try topk(@import("std").testing.allocator, .metal, .f32, input, values, indices, &.{ 2, 3 }, 1, 2, true);

    var v = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&v));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 9), v[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), v[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 10), v[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), v[3], 1e-5);

    var idx = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[1]);
    try @import("std").testing.expectEqual(@as(i64, 0), idx[2]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[3]);
}

test "metal topk supports smallest-k natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const values = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 9, 3, 10, 7, 6 }));
    try topk(@import("std").testing.allocator, .metal, .f32, input, values, indices, &.{ 2, 3 }, 1, 2, false);

    var v = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&v));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), v[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), v[1], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), v[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), v[3], 1e-5);

    var idx = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 0), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[1]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[2]);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[3]);
}

test "metal topk supports rank-1 vector" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 5 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const values = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 7, 1, 9, 3, 5 }));
    try topk(@import("std").testing.allocator, .metal, .f32, input, values, indices, &.{5}, 0, 2, true);

    var v = [_]f32{ 0, 0 };
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&v));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 9), v[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), v[1], 1e-5);

    var idx = [_]i64{ 0, 0 };
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectEqual(@as(i64, 2), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 0), idx[1]);
}

test "metal topk supports rank-3 axis natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const values = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();

    // shape [2,1,4], axis=2, k=2
    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{
        1, 9, 3, 7,
        8, 2, 6, 4,
    }));

    try topk(@import("std").testing.allocator, .metal, .f32, input, values, indices, &.{ 2, 1, 4 }, 2, 2, true);
    var v = [_]f32{ 0, 0, 0, 0 };
    var idx = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&v));
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 9), v[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 7), v[1], 1e-5);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 3), idx[1]);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 8), v[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 6), v[3], 1e-5);
    try @import("std").testing.expectEqual(@as(i64, 0), idx[2]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[3]);

    try topk(@import("std").testing.allocator, .metal, .f32, input, values, indices, &.{ 2, 1, 4 }, 2, 2, false);
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&v));
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&idx));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 1), v[0], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 3), v[1], 1e-5);
    try @import("std").testing.expectEqual(@as(i64, 0), idx[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), idx[1]);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 2), v[2], 1e-5);
    try @import("std").testing.expectApproxEqAbs(@as(f32, 4), v[3], 1e-5);
    try @import("std").testing.expectEqual(@as(i64, 1), idx[2]);
    try @import("std").testing.expectEqual(@as(i64, 3), idx[3]);
}

test "metal topk supports i64 input values natively" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const values = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer values.release();
    const indices = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer indices.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 9, 1, 7, 2, 8, 3 }));
    try topk(@import("std").testing.allocator, .metal, .i64, input, values, indices, &.{ 2, 3 }, 1, 2, true);

    var vv = [_]i64{ 0, 0, 0, 0 };
    var ii = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try values.metalHandle(), @import("std").mem.asBytes(&vv));
    try metal.common.readBuffer(try indices.metalHandle(), @import("std").mem.asBytes(&ii));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 9, 7, 8, 3 }, &vv);
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 0, 2, 1, 2 }, &ii);
}

test "metal gather uses native i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const index = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4, 5, 6 }));
    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 2, 1, 0, 2, 1, 0 }));
    try gather(.metal, .f32, input, index, out, &.{ 2, 3 }, &.{ 2, 3 }, 1);

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 3), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 1), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 6), ov[3]);
    try @import("std").testing.expectEqual(@as(f32, 5), ov[4]);
    try @import("std").testing.expectEqual(@as(f32, 4), ov[5]);
}

test "metal gather supports i64 values with i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const index = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 11, 12, 13, 21, 22, 23 }));
    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 2, 1, 0, 2, 1, 0 }));
    try gather(.metal, .i64, input, index, out, &.{ 2, 3 }, &.{ 2, 3 }, 1);

    var ov = [_]i64{ 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 13, 12, 11, 23, 22, 21 }, &ov);
}

test "metal index_select axis0 uses native i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const index = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4, 5, 6 }));
    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 0 }));
    try indexSelect(.metal, .f32, input, index, out, &.{ 2, 3 }, 0, 2);

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 4), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 5), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 6), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 1), ov[3]);
    try @import("std").testing.expectEqual(@as(f32, 2), ov[4]);
    try @import("std").testing.expectEqual(@as(f32, 3), ov[5]);
}

test "metal index_select supports axis1 with native i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const index = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4, 5, 6 }));
    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 2, 0 }));
    try indexSelect(.metal, .f32, input, index, out, &.{ 2, 3 }, 1, 2);

    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 3), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 1), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 6), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 4), ov[3]);
}

test "metal index_select supports i64 values with native i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const index = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 10, 20, 30, 40, 50, 60 }));
    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 2, 0 }));
    try indexSelect(.metal, .i64, input, index, out, &.{ 2, 3 }, 1, 2);

    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 30, 10, 60, 40 }, &ov);
}

test "metal one_hot uses native i64 index path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const index = try Storage.createMetalWithMetadata(allocator, 3 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 12 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 0, 2, 1 }));
    try oneHot(.metal, index, out, 4);

    var ov = [_]f32{0} ** 12;
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 1), ov[6]);
    try @import("std").testing.expectEqual(@as(f32, 1), ov[9]);
}

test "metal one_hot fails on out of bounds index" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const index = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer index.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try index.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 0, 7 }));
    try @import("std").testing.expectError(error.MetalKernelLaunchFailed, oneHot(.metal, index, out, 4));
}

test "metal masked_fill uses native i64 mask path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const mask = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mask.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try mask.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 0, 1, 0, 1 }));
    try maskedFill(.metal, .f32, .i64, input, mask, out, -9.0);

    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, -9), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 3), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, -9), ov[3]);
}

test "metal masked_fill supports i64 input/output" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const mask = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer mask.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try mask.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 0, 1, 0 }));
    try maskedFill(.metal, .i64, .i64, input, mask, out, -7.0);

    var ov = [_]i64{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, -7), ov[0]);
    try @import("std").testing.expectEqual(@as(i64, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(i64, -7), ov[2]);
    try @import("std").testing.expectEqual(@as(i64, 4), ov[3]);
}

test "metal dot uses native kernel path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 5, 6, 7, 8 }));
    try dot(.metal, .f32, a, b, out);

    var ov = [_]f32{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectApproxEqAbs(@as(f32, 70), ov[0], 1e-5);
}

test "metal dot supports i64" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 5, 6, 7, 8 }));
    try dot(.metal, .i64, a, b, out);

    var ov = [_]i64{0};
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(i64, 70), ov[0]);
}

test "metal contiguous uses native kernel path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 10, 20, 30, 40 }));
    try contiguous(.metal, .f32, input, out, &.{4}, &.{1}, 0);

    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 10), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 20), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 30), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 40), ov[3]);
}

test "metal cat uses native device copy path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 5, 6, 7, 8 }));
    const inputs = [_]*const Storage{ a, b };
    try cat(.metal, .f32, &inputs, out, &.{ 2, 2 }, 0);

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 3), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 4), ov[3]);
    try @import("std").testing.expectEqual(@as(f32, 5), ov[4]);
    try @import("std").testing.expectEqual(@as(f32, 6), ov[5]);
    try @import("std").testing.expectEqual(@as(f32, 7), ov[6]);
    try @import("std").testing.expectEqual(@as(f32, 8), ov[7]);
}

test "metal stack uses native device copy path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const a = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer a.release();
    const b = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer b.release();
    const out = try Storage.createMetalWithMetadata(allocator, 8 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try a.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try metal.common.writeBuffer(try b.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 5, 6, 7, 8 }));
    const inputs = [_]*const Storage{ a, b };
    try stack(.metal, .f32, &inputs, out, &.{ 2, 2 }, 1);

    var ov = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 1), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 2), ov[1]);
    try @import("std").testing.expectEqual(@as(f32, 5), ov[2]);
    try @import("std").testing.expectEqual(@as(f32, 6), ov[3]);
    try @import("std").testing.expectEqual(@as(f32, 3), ov[4]);
    try @import("std").testing.expectEqual(@as(f32, 4), ov[5]);
    try @import("std").testing.expectEqual(@as(f32, 7), ov[6]);
    try @import("std").testing.expectEqual(@as(f32, 8), ov[7]);
}

test "metal slice uses native strided contiguous path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4, 5, 6 }));
    const ranges = [_]@import("../types/operation/options.zig").SliceRange{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 1, .stop = 2, .step = 1 },
    };
    try slice(.metal, .f32, input, out, &.{ 2, 3 }, &ranges);

    var ov = [_]f32{ 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqual(@as(f32, 2), ov[0]);
    try @import("std").testing.expectEqual(@as(f32, 5), ov[1]);
}

test "metal slice supports i64 values with native strided contiguous path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 6 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 2 * @sizeOf(i64), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]i64{ 1, 2, 3, 4, 5, 6 }));
    const ranges = [_]@import("../types/operation/options.zig").SliceRange{
        .{ .start = 0, .stop = 2, .step = 1 },
        .{ .start = 1, .stop = 2, .step = 1 },
    };
    try slice(.metal, .i64, input, out, &.{ 2, 3 }, &ranges);

    var ov = [_]i64{ 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(i64, &[_]i64{ 2, 5 }, &ov);
}

test "metal contiguous supports negative strides with native kernel path" {
    if (!metal.common.isAvailable()) return error.SkipZigTest;

    const allocator = @import("std").testing.allocator;
    const input = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer input.release();
    const out = try Storage.createMetalWithMetadata(allocator, 4 * @sizeOf(f32), .{
        .policy = .pooled,
        .reason = .op_output,
        .source = .eager,
    });
    defer out.release();

    try metal.common.writeBuffer(try input.metalHandle(), @import("std").mem.asBytes(&[_]f32{ 1, 2, 3, 4 }));
    try contiguous(.metal, .f32, input, out, &.{4}, &.{-1}, 3);

    var ov = [_]f32{ 0, 0, 0, 0 };
    try metal.common.readBuffer(try out.metalHandle(), @import("std").mem.asBytes(&ov));
    try @import("std").testing.expectEqualSlices(f32, &[_]f32{ 4, 3, 2, 1 }, &ov);
}
