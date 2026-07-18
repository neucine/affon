pub const runner = @import("runner.zig");
pub const pir = @import("../../ir/pir/eager.zig");
pub const ExecutionResult = runner.ExecutionResult;
pub const EagerOpPlan = pir.Plan;
pub const execute = runner.execute;
pub const executeAll = runner.executeAll;
pub const executeAllWithPlan = runner.executeAllWithPlan;

test {
    _ = @import("runner.zig");
    _ = pir;
}
