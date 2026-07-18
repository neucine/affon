const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;

pub fn run(dtype: DType, input: *const Storage, out: *Storage) !void {
    switch (dtype) {
        .f32 => {
            const src = std.mem.bytesAsSlice(f32, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f32, try out.writableBytes());
            const n = @as(f32, @floatFromInt(src.len));
            var mean: f32 = 0;
            for (src) |v| mean += v;
            mean /= n;
            var var_acc: f32 = 0;
            for (src) |v| {
                const d = v - mean;
                var_acc += d * d;
            }
            dst[0] = var_acc / n;
        },
        .f64 => {
            const src = std.mem.bytesAsSlice(f64, try input.readableBytes());
            const dst = std.mem.bytesAsSlice(f64, try out.writableBytes());
            const n = @as(f64, @floatFromInt(src.len));
            var mean: f64 = 0;
            for (src) |v| mean += v;
            mean /= n;
            var var_acc: f64 = 0;
            for (src) |v| {
                const d = v - mean;
                var_acc += d * d;
            }
            dst[0] = var_acc / n;
        },
        .i64 => return error.ExecutionNotImplemented,
    }
}
