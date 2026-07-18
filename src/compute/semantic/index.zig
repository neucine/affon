const spec = @import("spec.zig");
const infer_mod = @import("infer.zig");

pub const ExecutionKind = spec.ExecutionKind;
pub const AllocationIntent = spec.AllocationIntent;
pub const InputRequirement = spec.InputRequirement;
pub const OutputSpec = spec.OutputSpec;
pub const BinaryBroadcastSpec = spec.BinaryBroadcastSpec;
pub const WhereBroadcastSpec = spec.WhereBroadcastSpec;
pub const MaskedFillBroadcastSpec = spec.MaskedFillBroadcastSpec;
pub const ReduceToShapeSpec = spec.ReduceToShapeSpec;
pub const BroadcastSpec = spec.BroadcastSpec;
pub const PlannerHint = spec.PlannerHint;
pub const OpSpec = spec.OpSpec;

pub const infer = infer_mod.infer;
pub const inferFromSpecs = infer_mod.inferFromSpecs;

test {
    _ = spec;
    _ = infer_mod;
}
