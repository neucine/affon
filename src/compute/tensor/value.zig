const std = @import("std");
const Device = @import("device.zig").Device;
const DType = @import("dtype.zig").DType;
const Shape = @import("shape.zig").Shape;
const Layout = @import("layout.zig").Layout;
const Storage = @import("storage.zig").Storage;
const ValueSpec = @import("value_spec.zig").ValueSpec;

pub const AxisName = @import("axis.zig").AxisName;

pub fn cloneAxes(allocator: std.mem.Allocator, axes: ?[]const AxisName) !?[]const AxisName {
    const source = axes orelse return null;
    const out = try allocator.alloc(AxisName, source.len);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |axis| allocator.free(axis);
        allocator.free(out);
    }
    for (source, 0..) |axis, i| {
        out[i] = try allocator.dupe(u8, axis);
        initialized += 1;
    }
    return out;
}

pub fn deinitAxes(allocator: std.mem.Allocator, axes: ?[]const AxisName) void {
    const names = axes orelse return;
    for (names) |axis| allocator.free(axis);
    allocator.free(names);
}

pub const Value = struct {
    // Logical tensor facts. These are the facts semantic analysis and planning
    // should reason about through ValueSpec when possible.
    allocator: std.mem.Allocator,
    shape: Shape,
    dtype: DType,
    layout: Layout,
    axes: ?[]const AxisName = null,

    // Runtime backing. Execution and kernels may require this; semantic and
    // planning layers should prefer ValueSpec and avoid interpreting storage.
    storage: ?*Storage,

    // Internal optional sidecar for autograd provenance / retained grad state.
    autograd_state: ?*anyopaque = null,

    pub fn createContiguous(
        allocator: std.mem.Allocator,
        dims: []const usize,
        dtype: DType,
        target_device: Device,
        zeroed: bool,
    ) !*Value {
        return createContiguousWithSource(allocator, dims, dtype, target_device, zeroed, .eager);
    }

    pub fn createContiguousWithSource(
        allocator: std.mem.Allocator,
        dims: []const usize,
        dtype: DType,
        target_device: Device,
        zeroed: bool,
        source: Storage.Source,
    ) !*Value {
        var shape = try Shape.initCopy(allocator, dims);
        errdefer shape.deinit();
        var layout = try Layout.initContiguous(allocator, shape);
        errdefer layout.deinit();

        const storage = switch (target_device) {
            .cpu => try Storage.createCpuWithMetadata(allocator, shape.numel() * dtype.size(), zeroed, .{
                .reason = .op_output,
                .source = source,
            }),
            .metal => try Storage.createMetalWithMetadata(allocator, shape.numel() * dtype.size(), .{
                .policy = .pooled,
                .reason = .op_output,
                .source = source,
            }),
        };
        errdefer storage.release();

        const self = try allocator.create(Value);
        self.* = .{
            .allocator = allocator,
            .shape = shape,
            .dtype = dtype,
            .layout = layout,
            .storage = storage,
            .axes = null,
            .autograd_state = null,
        };
        return self;
    }

    pub fn fromSliceF64(allocator: std.mem.Allocator, dims: []const usize, values: []const f64) !*Value {
        const self = try createContiguous(allocator, dims, .f64, .cpu, false);
        errdefer self.deinit();
        if (self.shape.numel() != values.len) return error.SizeMismatch;
        const bytes = try self.storage.?.writableBytes();
        const typed: []f64 = std.mem.bytesAsSlice(f64, bytes);
        @memcpy(typed, values);
        return self;
    }

    pub fn fromSliceF32(allocator: std.mem.Allocator, dims: []const usize, values: []const f32) !*Value {
        const self = try createContiguous(allocator, dims, .f32, .cpu, false);
        errdefer self.deinit();
        if (self.shape.numel() != values.len) return error.SizeMismatch;
        const bytes = try self.storage.?.writableBytes();
        const typed: []f32 = std.mem.bytesAsSlice(f32, bytes);
        @memcpy(typed, values);
        return self;
    }

    pub fn fromSliceI64(allocator: std.mem.Allocator, dims: []const usize, values: []const i64) !*Value {
        const self = try createContiguous(allocator, dims, .i64, .cpu, false);
        errdefer self.deinit();
        if (self.shape.numel() != values.len) return error.SizeMismatch;
        const bytes = try self.storage.?.writableBytes();
        const typed: []i64 = std.mem.bytesAsSlice(i64, bytes);
        @memcpy(typed, values);
        return self;
    }

    pub fn deinit(self: *Value) void {
        if (self.storage) |storage| storage.release();
        deinitAxes(self.allocator, self.axes);
        self.layout.deinit();
        self.shape.deinit();
        self.allocator.destroy(self);
    }

    pub fn setAxesCopy(self: *Value, axes: ?[]const AxisName) !void {
        if (axes) |names| {
            if (names.len != self.shape.rank()) return error.AxisRankMismatch;
        }
        const cloned = try cloneAxes(self.allocator, axes);
        deinitAxes(self.allocator, self.axes);
        self.axes = cloned;
    }

    pub fn device(self: *const Value) ?Device {
        const storage = self.storage orelse return null;
        return storage.device();
    }

    pub fn runtimeBacking(self: *const Value) ?*Storage {
        return self.storage;
    }

    pub fn requireRuntimeBacking(self: *const Value) !*Storage {
        return self.runtimeBacking() orelse error.InputNotMaterialized;
    }

    pub fn spec(self: *const Value) !ValueSpec {
        const target_device = self.device() orelse return error.InputNotMaterialized;
        return .{
            .shape = self.shape,
            .dtype = self.dtype,
            .layout = self.layout,
            .device = target_device,
            .axes = self.axes,
        };
    }

    pub fn autogradStateRaw(self: *const Value) ?*anyopaque {
        return self.autograd_state;
    }

    pub fn setAutogradStateRaw(self: *Value, autograd_state: ?*anyopaque) void {
        self.autograd_state = autograd_state;
    }
};

test "value from slice f64 copies host data" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceF64(allocator, &.{2}, &.{ 1.0, 2.0 });
    defer value.deinit();

    const bytes = try value.storage.?.readableBytes();
    const typed: []align(@alignOf(f64)) const f64 = std.mem.bytesAsSlice(f64, bytes);
    try std.testing.expectEqual(@as(f64, 2.0), typed[1]);
}

test "value from slice i64 copies host data" {
    const allocator = std.testing.allocator;
    const value = try Value.fromSliceI64(allocator, &.{3}, &.{ 4, 5, 6 });
    defer value.deinit();

    const bytes = try value.storage.?.readableBytes();
    const typed: []align(@alignOf(i64)) const i64 = std.mem.bytesAsSlice(i64, bytes);
    try std.testing.expectEqual(@as(i64, 5), typed[1]);
}
