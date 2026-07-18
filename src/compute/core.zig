pub const tensor = @import("tensor/index.zig");
pub const op = @import("op/index.zig");
pub const cpu = @import("kernel/cpu/index.zig");
pub const execution = @import("execution/core.zig");

test {
    _ = tensor;
    _ = op;
    _ = cpu;
    _ = execution;
}
