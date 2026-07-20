pub const types = @import("types/index.zig");
pub const compose = @import("compose/index.zig");
pub const tensor = types.tensor;
pub const operation = types.operation;
pub const ir = types.ir;
pub const backend = @import("backend/index.zig");
pub const execution = @import("execution/core.zig");
pub const eager = execution.eager;
pub const graph = execution.graph;
pub const Engine = @import("engine.zig").Engine;
pub const telemetry = @import("telemetry.zig");
pub const telemetry_types = telemetry;

test {
    _ = tensor;
    _ = operation;
    _ = ir;
    _ = backend;
    _ = execution;
    _ = eager;
    _ = graph;
}
