const std = @import("std");

var enabled = std.atomic.Value(bool).init(true);

pub fn isEnabled() bool {
    return enabled.load(.monotonic);
}

pub fn setEnabled(value: bool) void {
    enabled.store(value, .monotonic);
}
