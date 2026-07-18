const std = @import("std");
const tensor_value = @import("../../tensor/value.zig");
const Device = @import("../../tensor/device.zig").Device;
const DType = @import("../../tensor/dtype.zig").DType;
const Layout = @import("../../tensor/layout.zig").Layout;
const Shape = @import("../../tensor/shape.zig").Shape;
const OpTag = @import("../../op/tag.zig").OpTag;
const semantic = @import("../../semantic/index.zig");
const execution_layout = @import("../../execution/layout.zig");
const execution_spec = @import("../../execution/spec.zig");

pub const ExecutionInputRequirement = execution_spec.InputRequirement;

pub const Plan = struct {
    pub const Input = struct {
        dtype: DType,
        device: Device,
        shape: Shape,

        pub fn deinit(self: *Input) void {
            self.shape.deinit();
            self.* = undefined;
        }
    };

    pub const Output = struct {
        dtype: DType,
        shape: Shape,
        layout: Layout,
        axes: ?[]const @import("../../tensor/axis.zig").AxisName = null,
        bytes: usize,

        pub fn deinit(self: *Output) void {
            tensor_value.deinitAxes(self.shape.allocator, self.axes);
            self.layout.deinit();
            self.shape.deinit();
            self.* = undefined;
        }
    };

    op_tag: OpTag,
    allocator: std.mem.Allocator,
    device: Device,
    kind: execution_spec.ExecutionKind,
    input_requirement: ExecutionInputRequirement,
    input_layout_decision: execution_layout.InputLayoutDecision,
    broadcast: ?semantic.BroadcastSpec,
    reduce_to_shape: ?semantic.ReduceToShapeSpec,
    inputs: []Input,
    primary_output: Output,
    secondary_output: ?Output,

    pub fn deinit(self: *Plan) void {
        for (self.inputs) |*input| input.deinit();
        self.allocator.free(self.inputs);
        if (self.secondary_output) |*secondary| secondary.deinit();
        self.primary_output.deinit();
        self.* = undefined;
    }
};
