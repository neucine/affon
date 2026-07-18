const semantic = @import("../sema/index.zig");
const kernel_capability = @import("../backend/capability.zig");

pub const InputLayoutDecision = kernel_capability.InputLayoutDecision;

pub fn inputDecisionForHint(hint: ?semantic.PlannerHint) InputLayoutDecision {
    return if (hint) |planner_hint| planner_hint.input_layout_decision else .accept;
}
