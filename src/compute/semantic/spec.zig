const tensor = @import("../tensor/index.zig");
const Shape = tensor.Shape;
const Layout = tensor.Layout;
const Device = tensor.Device;
const DType = tensor.DType;
const AxisName = tensor.AxisName;
const kernel_capability = @import("../kernel/capability.zig");

pub const ExecutionKind = enum {
    elementwise_binary,
    elementwise_unary,
    elementwise_generic,
    reduction_all,
    reduction,
    index,
    view,
};

pub const AllocationIntent = enum {
    new_storage,
    view_only,
};

pub const InputRequirement = enum {
    preserve,
    require_storage,
    require_contiguous_input,
};

pub const OutputSpec = struct {
    shape: Shape,
    dtype: DType,
    layout: Layout,
    axes: ?[]const AxisName = null,

    pub fn deinit(self: *OutputSpec) void {
        tensor.deinitAxes(self.shape.allocator, self.axes);
        self.layout.deinit();
        self.shape.deinit();
        self.* = undefined;
    }
};

pub const BinaryBroadcastSpec = struct {
    rank: usize,
    shape: [8]usize,
    lhs_strides: [8]isize,
    rhs_strides: [8]isize,
};

pub const WhereBroadcastSpec = struct {
    rank: usize,
    shape: [8]usize,
    cond_strides: [8]isize,
    on_true_strides: [8]isize,
    on_false_strides: [8]isize,
};

pub const MaskedFillBroadcastSpec = struct {
    rank: usize,
    shape: [8]usize,
    input_strides: [8]isize,
    mask_strides: [8]isize,
};

pub const ReduceToShapeSpec = struct {
    axes: [8]usize,
    count: usize,
};

pub const BroadcastSpec = union(enum) {
    binary: BinaryBroadcastSpec,
    where: WhereBroadcastSpec,
    masked_fill: MaskedFillBroadcastSpec,
};

pub const PlannerHint = struct {
    input_layout_decision: kernel_capability.InputLayoutDecision = .accept,
};

pub const OpSpec = struct {
    shape: Shape,
    dtype: DType,
    layout: Layout,
    device: Device,
    kind: ExecutionKind,
    allocation: AllocationIntent,
    input_requirement: InputRequirement,
    broadcast: ?BroadcastSpec = null,
    reduce_to_shape: ?ReduceToShapeSpec = null,
    planner_hint: ?PlannerHint = null,
    secondary_output: ?OutputSpec = null,
    axes: ?[]const AxisName = null,

    pub fn deinit(self: *OpSpec) void {
        if (self.secondary_output) |*secondary| secondary.deinit();
        tensor.deinitAxes(self.shape.allocator, self.axes);
        self.layout.deinit();
        self.shape.deinit();
        self.* = undefined;
    }

    pub fn clone(allocator: @import("std").mem.Allocator, source: OpSpec) !OpSpec {
        var shape = try Shape.initCopy(allocator, source.shape.dims);
        errdefer shape.deinit();
        var layout = try Layout.initCopy(allocator, source.layout.strides, source.layout.offset);
        errdefer layout.deinit();
        const axes = try tensor.cloneAxes(allocator, source.axes);
        errdefer tensor.deinitAxes(allocator, axes);

        var secondary_output: ?OutputSpec = null;
        if (source.secondary_output) |secondary| {
            var secondary_shape = try Shape.initCopy(allocator, secondary.shape.dims);
            errdefer secondary_shape.deinit();
            var secondary_layout = try Layout.initCopy(allocator, secondary.layout.strides, secondary.layout.offset);
            errdefer secondary_layout.deinit();
            const secondary_axes = try tensor.cloneAxes(allocator, secondary.axes);
            errdefer tensor.deinitAxes(allocator, secondary_axes);
            secondary_output = .{
                .shape = secondary_shape,
                .dtype = secondary.dtype,
                .layout = secondary_layout,
                .axes = secondary_axes,
            };
        }

        return .{
            .shape = shape,
            .dtype = source.dtype,
            .layout = layout,
            .device = source.device,
            .kind = source.kind,
            .allocation = source.allocation,
            .input_requirement = source.input_requirement,
            .broadcast = source.broadcast,
            .reduce_to_shape = source.reduce_to_shape,
            .planner_hint = source.planner_hint,
            .secondary_output = secondary_output,
            .axes = axes,
        };
    }
};
