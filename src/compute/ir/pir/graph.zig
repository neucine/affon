const std = @import("std");
const graph_mod = @import("../sir.zig");
const semantic = @import("../../semantic/index.zig");
const execution_layout = @import("../../execution/layout.zig");
const execution_spec = @import("../../execution/spec.zig");

pub const Decisions = struct {
    kind: execution_spec.ExecutionKind,
    allocation: execution_spec.AllocationIntent,
    input_requirement: execution_spec.InputRequirement,
    input_layout_decision: execution_layout.InputLayoutDecision,
    broadcast: ?semantic.BroadcastSpec,
    reduce_to_shape: ?semantic.ReduceToShapeSpec,
};

pub const StepKind = enum {
    single_op,
    fused_elementwise_region,
};

pub const Step = struct {
    node_id: graph_mod.NodeId,
    kind: StepKind,
    /// Semantic evidence captured during graph lowering. Runtime execution may
    /// consume these facts, but they are not themselves backend dispatch plans.
    semantic_spec: semantic.OpSpec,
    /// Executable decisions derived from semantic evidence for graph runtime.
    /// This is the boundary where lowering starts to become a runtime plan.
    execution: Decisions,

    pub fn deinit(self: *Step) void {
        self.semantic_spec.deinit();
        self.* = undefined;
    }
};

pub const RegionKind = enum {
    fusable_run,
    matmul_epilogue,
};

pub const MatmulEpilogueActivation = enum {
    none,
    gelu,
    relu,
    sigmoid,
};

pub const Region = struct {
    kind: RegionKind,
    step_start: usize,
    step_end: usize,
    matmul_epilogue_activation: ?MatmulEpilogueActivation = null,
};

pub const Plan = struct {
    allocator: std.mem.Allocator,
    steps: std.ArrayList(Step),
    regions: std.ArrayList(Region),
    outputs: std.ArrayList(graph_mod.ValueId),

    pub fn init(allocator: std.mem.Allocator) Plan {
        return .{
            .allocator = allocator,
            .steps = .empty,
            .regions = .empty,
            .outputs = .empty,
        };
    }

    pub fn deinit(self: *Plan) void {
        self.outputs.deinit(self.allocator);
        self.regions.deinit(self.allocator);
        for (self.steps.items) |*step| step.deinit();
        self.steps.deinit(self.allocator);
        self.* = undefined;
    }
};
