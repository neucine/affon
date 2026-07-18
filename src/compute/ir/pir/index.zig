pub const eager = @import("eager.zig");
pub const graph = @import("graph.zig");

pub const EagerPlan = eager.Plan;
pub const GraphPlan = graph.Plan;

test {
    _ = eager;
    _ = graph;
}
