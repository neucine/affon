const std = @import("std");
const sir = @import("../sir.zig");
const pir = @import("../pir/graph.zig");
const semantic = @import("../../../sema/index.zig");
const execution_layout = @import("../../../execution/layout.zig");
const execution_spec = @import("../../../execution/spec.zig");

pub const RegionKind = pir.RegionKind;
pub const MatmulEpilogueActivation = pir.MatmulEpilogueActivation;
pub const Region = pir.Region;

pub const Decisions = struct {
    kind: execution_spec.ExecutionKind,
    allocation: execution_spec.AllocationIntent,
    input_requirement: execution_spec.InputRequirement,
    input_layout_decision: execution_layout.InputLayoutDecision,
    broadcast: ?semantic.BroadcastSpec,
    reduce_to_shape: ?semantic.ReduceToShapeSpec,
};

pub const Step = struct {
    node_id: sir.NodeId,
    kind: pir.StepKind,
    semantic_spec: semantic.OpSpec,
    execution: Decisions,

    pub fn deinit(self: *Step) void {
        self.semantic_spec.deinit();
        self.* = undefined;
    }
};

pub const Program = struct {
    allocator: std.mem.Allocator,
    steps: std.ArrayList(Step),
    regions: std.ArrayList(pir.Region),
    outputs: std.ArrayList(sir.ValueId),

    pub fn init(allocator: std.mem.Allocator) Program {
        return .{
            .allocator = allocator,
            .steps = .empty,
            .regions = .empty,
            .outputs = .empty,
        };
    }

    pub fn deinit(self: *Program) void {
        self.outputs.deinit(self.allocator);
        self.regions.deinit(self.allocator);
        for (self.steps.items) |*step| step.deinit();
        self.steps.deinit(self.allocator);
        self.* = undefined;
    }
};
