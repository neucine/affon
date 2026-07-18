const std = @import("std");
const graph_mod = @import("graph.zig");
const tensor = @import("../tensor/tensor.zig");
const Device = @import("../tensor/device.zig").Device;
const DType = @import("../tensor/dtype.zig").DType;
const Layout = @import("../tensor/layout.zig").Layout;
const Shape = @import("../tensor/shape.zig").Shape;
const OpTag = @import("../operation/tag.zig").OpTag;
const semantic = @import("../../plan/sema/index.zig");
const execution_layout = @import("../../plan/layout.zig");
const execution_spec = @import("../../plan/spec.zig");

pub const ExecutionInputRequirement = execution_spec.InputRequirement;
pub const ReduceToShapeSpec = semantic.ReduceToShapeSpec;
pub const BroadcastSpec = semantic.BroadcastSpec;
pub const BinaryBroadcastSpec = semantic.BinaryBroadcastSpec;
pub const MaskedFillBroadcastSpec = semantic.MaskedFillBroadcastSpec;
pub const WhereBroadcastSpec = semantic.WhereBroadcastSpec;

pub const MatmulFamily = enum {
    gemm_2d,
    gemm_batched,
    gemm_projection,
    gemm_attention_scores,
    gemm_attention_values,
    gemm_generic_unresolved,
};

pub const MatmulDecision = struct {
    family: MatmulFamily,
    projection: bool,
};

pub const EagerPlan = struct {
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
        axes: ?[]const @import("../tensor/axis.zig").AxisName = null,
        bytes: usize,

        pub fn deinit(self: *Output) void {
            tensor.deinitAxes(self.shape.allocator, self.axes);
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
    matmul: ?MatmulDecision,
    inputs: []Input,
    primary_output: Output,
    secondary_output: ?Output,

    pub fn deinit(self: *EagerPlan) void {
        for (self.inputs) |*input| input.deinit();
        self.allocator.free(self.inputs);
        if (self.secondary_output) |*secondary| secondary.deinit();
        self.primary_output.deinit();
        self.* = undefined;
    }
};

pub const Decisions = struct {
    kind: execution_spec.ExecutionKind,
    allocation: execution_spec.AllocationIntent,
    input_requirement: execution_spec.InputRequirement,
    input_layout_decision: execution_layout.InputLayoutDecision,
    broadcast: ?semantic.BroadcastSpec,
    reduce_to_shape: ?semantic.ReduceToShapeSpec,
};

pub const StepKind = enum { single_op, fused_elementwise_region };

pub const Step = struct {
    node_id: graph_mod.NodeId,
    kind: StepKind,
    semantic_spec: semantic.OpSpec,
    eager_plan: EagerPlan,
    execution: Decisions,

    pub fn deinit(self: *Step) void {
        self.eager_plan.deinit();
        self.semantic_spec.deinit();
        self.* = undefined;
    }
};

pub const RegionKind = enum { fusable_run, matmul_epilogue };
pub const MatmulEpilogueActivation = enum { none, gelu, relu, sigmoid };

pub const Region = struct {
    kind: RegionKind,
    step_start: usize,
    step_end: usize,
    matmul_epilogue_activation: ?MatmulEpilogueActivation = null,
};

pub const GraphPlan = struct {
    allocator: std.mem.Allocator,
    steps: std.ArrayList(Step),
    regions: std.ArrayList(Region),
    outputs: std.ArrayList(graph_mod.TensorId),

    pub fn init(allocator: std.mem.Allocator) GraphPlan {
        return .{ .allocator = allocator, .steps = .empty, .regions = .empty, .outputs = .empty };
    }

    pub fn deinit(self: *GraphPlan) void {
        self.outputs.deinit(self.allocator);
        self.regions.deinit(self.allocator);
        for (self.steps.items) |*step| step.deinit();
        self.steps.deinit(self.allocator);
        self.* = undefined;
    }
};
