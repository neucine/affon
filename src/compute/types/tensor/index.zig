pub const Device = @import("device.zig").Device;
pub const DType = @import("dtype.zig").DType;
pub const AxisName = @import("axis.zig").AxisName;
pub const Shape = @import("shape.zig").Shape;
pub const Layout = @import("layout.zig").Layout;
pub const Storage = @import("storage.zig").Storage;
pub const RuntimeBacking = Storage;
pub const Value = @import("value.zig").Value;
pub const ValueSpec = @import("value_spec.zig").ValueSpec;
pub const cloneAxes = @import("value.zig").cloneAxes;
pub const deinitAxes = @import("value.zig").deinitAxes;

test {
    _ = Device;
    _ = DType;
    _ = AxisName;
    _ = Shape;
    _ = Layout;
    _ = Storage;
    _ = RuntimeBacking;
    _ = Value;
    _ = ValueSpec;
    _ = cloneAxes;
    _ = deinitAxes;
}
