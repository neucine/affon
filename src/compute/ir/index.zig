pub const sir = @import("sir.zig");
pub const pir = @import("pir/index.zig");
pub const eir = @import("eir/index.zig");

pub const ValueId = sir.ValueId;
pub const NodeId = sir.NodeId;
pub const NodeKind = sir.NodeKind;
pub const Node = sir.Node;
pub const GraphValue = sir.GraphValue;
pub const Graph = sir.Graph;

test {
    _ = sir;
    _ = pir;
    _ = eir;
}
