const std = @import("std");
const DType = @import("../../tensor/dtype.zig").DType;
const Storage = @import("../../tensor/storage.zig").Storage;
const OpTag = @import("../../op/tag.zig").OpTag;

pub const UnaryStage = struct {
    tag: OpTag,
    clamp_min: ?f64 = null,
    clamp_max: ?f64 = null,
};

pub fn unaryChain(dtype: DType, input: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    if (stages.len == 0) return error.InvalidInputCount;
    switch (dtype) {
        .f32 => try unaryChainTyped(f32, input, out, stages),
        .f64 => try unaryChainTyped(f64, input, out, stages),
        .i64 => try unaryChainTyped(i64, input, out, stages),
    }
}

pub fn binaryThenUnaryChain(dtype: DType, binary_tag: OpTag, lhs: *const Storage, rhs: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    if (stages.len == 0) return error.InvalidInputCount;
    switch (dtype) {
        .f32 => try binaryThenUnaryChainTyped(f32, binary_tag, lhs, rhs, out, stages),
        .f64 => try binaryThenUnaryChainTyped(f64, binary_tag, lhs, rhs, out, stages),
        .i64 => try binaryThenUnaryChainTyped(i64, binary_tag, lhs, rhs, out, stages),
    }
}

fn unaryChainTyped(comptime T: type, input: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    const src = std.mem.bytesAsSlice(T, try input.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    for (dst, src) |*d, s| {
        var v = s;
        for (stages) |st| v = try applyUnary(T, st, v);
        d.* = v;
    }
}

fn binaryThenUnaryChainTyped(comptime T: type, binary_tag: OpTag, lhs: *const Storage, rhs: *const Storage, out: *Storage, stages: []const UnaryStage) !void {
    const av = std.mem.bytesAsSlice(T, try lhs.readableBytes());
    const bv = std.mem.bytesAsSlice(T, try rhs.readableBytes());
    const dst = std.mem.bytesAsSlice(T, try out.writableBytes());
    for (dst, 0..) |*d, i| {
        var v = try applyBinary(T, binary_tag, av[i], bv[i]);
        for (stages) |st| v = try applyUnary(T, st, v);
        d.* = v;
    }
}

fn applyBinary(comptime T: type, tag: OpTag, a: T, b: T) !T {
    return switch (tag) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => if (T == i64) @divTrunc(a, b) else a / b,
        else => error.ExecutionNotImplemented,
    };
}

fn applyUnary(comptime T: type, st: UnaryStage, x: T) !T {
    return switch (st.tag) {
        .abs => if (T == i64) (if (x < 0) -x else x) else @abs(x),
        .neg => -x,
        .relu => if (T == i64) @max(x, @as(i64, 0)) else @max(x, @as(T, 0)),
        .silu => if (T == i64) return error.ExecutionNotImplemented else x * (1.0 / (1.0 + @exp(-x))),
        .sign => blk: {
            if (x > 0) break :blk 1;
            if (x < 0) break :blk -1;
            break :blk 0;
        },
        .exp => if (T == i64) return error.ExecutionNotImplemented else @exp(x),
        .log => if (T == i64) return error.ExecutionNotImplemented else @log(x),
        .sqrt => if (T == i64) return error.ExecutionNotImplemented else @sqrt(x),
        .clamp => blk: {
            const lo = st.clamp_min orelse return error.InvalidClampBounds;
            const hi = st.clamp_max orelse return error.InvalidClampBounds;
            if (T == f32) break :blk @min(@max(x, @as(T, @floatCast(lo))), @as(T, @floatCast(hi)));
            if (T == f64) break :blk @min(@max(x, @as(T, lo)), @as(T, hi));
            if (@trunc(lo) != lo or @trunc(hi) != hi) return error.InvalidClampBounds;
            const loi: i64 = @intFromFloat(lo);
            const hii: i64 = @intFromFloat(hi);
            break :blk @min(@max(x, @as(T, loi)), @as(T, hii));
        },
        else => error.ExecutionNotImplemented,
    };
}
