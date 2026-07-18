pub const State = @import("state.zig").State;
pub const TrackingState = @import("state.zig").TrackingState;
pub const Node = @import("types.zig").Node;
pub const Parent = @import("types.zig").Parent;
pub const tape = @import("tape.zig");

pub fn makeTrainableValue(allocator: @import("std").mem.Allocator, value: *@import("../types/tensor/value.zig").Value) !*@import("../types/tensor/value.zig").Value {
    _ = try State.create(allocator, value, true);
    return value;
}

pub fn makeTrackedValue(allocator: @import("std").mem.Allocator, value: *@import("../types/tensor/value.zig").Value) !*@import("../types/tensor/value.zig").Value {
    _ = try State.create(allocator, value, false);
    return value;
}

pub fn releaseOwnedValue(value: *@import("../types/tensor/value.zig").Value) void {
    State.releaseOwnedValue(value);
}

test {
    _ = tape;
}
