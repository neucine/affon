const std = @import("std");
const Tensor = @import("../../../types/tensor/tensor.zig").Tensor;
const Shape = @import("../../../types/tensor/shape.zig").Shape;
const Layout = @import("../../../types/tensor/layout.zig").Layout;
const Device = @import("../../../types/tensor/device.zig").Device;

pub fn moveToDevice(allocator: std.mem.Allocator, value: *const Tensor, target: Device) !*Tensor {
    if ((value.device() orelse return error.InputNotMaterialized) != target) return error.DeviceMismatch;
    return cloneValue(allocator, value);
}

pub fn cloneValue(allocator: std.mem.Allocator, value: *const Tensor) !*Tensor {
    const storage = value.storage orelse return error.InputNotMaterialized;
    storage.retain();
    errdefer storage.release();
    const cloned = try allocator.create(Tensor);
    errdefer allocator.destroy(cloned);
    var shape = try Shape.initCopy(allocator, value.shape.dims);
    errdefer shape.deinit();
    var layout = try Layout.initCopy(allocator, value.layout.strides, value.layout.offset);
    errdefer layout.deinit();
    cloned.* = .{
        .allocator = allocator,
        .shape = shape,
        .dtype = value.dtype,
        .layout = layout,
        .storage = storage,
        .axes = null,
    };
    return cloned;
}
