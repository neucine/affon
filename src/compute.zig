const config = @import("config.zig");

pub const engine = @import("compute/engine.zig");
pub const compose = @import("compute/compose/index.zig");
pub const Engine = engine.Engine;
pub const Tensor = engine.Tensor;
pub const TensorSpec = @import("compute/types/tensor/index.zig").TensorSpec;
pub const ComputeGraph = @import("compute/types/ir/index.zig").ComputeGraph;
pub const Operation = engine.Operation;

comptime {
    _ = config;
}
