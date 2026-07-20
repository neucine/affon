pub const runner = @import("runner.zig");
pub const plan = @import("../../shared/types/ir/plan.zig");
pub const ExecutionResult = runner.ExecutionResult;
pub const EagerOpPlan = plan.EagerPlan;
pub const executeAllWithPlan = runner.executeAllWithPlan;

test {
    _ = @import("runner.zig");
    _ = plan;
}
