pub const tensor = @import("tensor/index.zig");
pub const operation = @import("operation/index.zig");
pub const ir = @import("ir/index.zig");

test {
    _ = tensor;
    _ = operation;
    _ = ir;
}
