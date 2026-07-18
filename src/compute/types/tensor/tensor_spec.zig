const Device = @import("device.zig").Device;
const DType = @import("dtype.zig").DType;
const Shape = @import("shape.zig").Shape;
const Layout = @import("layout.zig").Layout;
const AxisName = @import("axis.zig").AxisName;

pub const TensorSpec = struct {
    // Logical tensor facts accepted by semantic analysis and consumed by
    // planning. TensorSpec does not own or imply runtime storage.
    shape: Shape,
    dtype: DType,
    layout: Layout,
    device: Device,
    axes: ?[]const AxisName = null,
};
