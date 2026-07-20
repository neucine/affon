const std = @import("std");
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;

pub fn run(dtype: DType, logits: *const Storage, targets: *const Storage, out: *Storage, rows: usize, classes: usize) !void {
    if (rows == 0 or classes == 0) return error.ShapeMismatch;
    const target_ids = std.mem.bytesAsSlice(i64, try targets.readableBytes());
    if (target_ids.len != rows and target_ids.len != rows * 1) return error.ShapeMismatch;
    switch (dtype) {
        .f32 => try runTyped(f32, logits, target_ids, out, rows, classes),
        .f64 => try runTyped(f64, logits, target_ids, out, rows, classes),
        .i64 => return error.ExecutionNotImplemented,
    }
}

fn runTyped(comptime T: type, logits: *const Storage, targets: []const i64, out: *Storage, rows: usize, classes: usize) !void {
    const x = std.mem.bytesAsSlice(T, try logits.readableBytes());
    const y = std.mem.bytesAsSlice(T, try out.writableBytes());
    if (x.len != rows * classes) return error.ShapeMismatch;

    var acc: T = 0;
    for (0..rows) |row| {
        const target = targets[row];
        if (target < 0 or target >= classes) return error.IndexOutOfBounds;
        const target_index: usize = @intCast(target);
        const base = row * classes;

        var max_v = x[base];
        var c: usize = 1;
        while (c < classes) : (c += 1) {
            const v = x[base + c];
            if (v > max_v) max_v = v;
        }

        var sum_exp: T = 0;
        c = 0;
        while (c < classes) : (c += 1) {
            sum_exp += @exp(x[base + c] - max_v);
        }
        acc += @log(sum_exp) + max_v - x[base + target_index];
    }

    y[0] = acc / @as(T, @floatFromInt(rows));
}
