pub const eager = @import("eager/index.zig");
pub const graph = @import("graph/index.zig");
pub const autograd = @import("autograd.zig");

test {
    _ = eager;
    _ = graph;
    _ = autograd;
}
