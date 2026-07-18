pub const Device = @import("device.zig").Device;
pub const DType = @import("dtype.zig").DType;
pub const AxisName = @import("axis.zig").AxisName;
pub const Shape = @import("shape.zig").Shape;
pub const Layout = @import("layout.zig").Layout;
pub const Storage = @import("storage.zig").Storage;
pub const RuntimeBacking = Storage;
pub const Tensor = @import("tensor.zig").Tensor;
pub const TensorSpec = @import("tensor_spec.zig").TensorSpec;
pub const cloneAxes = @import("tensor.zig").cloneAxes;
pub const deinitAxes = @import("tensor.zig").deinitAxes;

test {
    _ = Device;
    _ = DType;
    _ = AxisName;
    _ = Shape;
    _ = Layout;
    _ = Storage;
    _ = RuntimeBacking;
    _ = Tensor;
    _ = TensorSpec;
    _ = cloneAxes;
    _ = deinitAxes;
}
