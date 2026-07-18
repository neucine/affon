const std = @import("std");

pub const Diagnostic = struct {
    const max_len = 1024;
    const truncation_marker = "... (message truncated)";

    message: [max_len]u8 = undefined,
    message_len: usize = 0,

    pub fn set(self: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        const result = std.fmt.bufPrint(&self.message, fmt, args) catch {
            const safe = max_len - truncation_marker.len;
            @memcpy(self.message[safe..], truncation_marker);
            self.message_len = max_len;
            return;
        };
        self.message_len = result.len;
    }

    pub fn slice(self: *const Diagnostic) []const u8 {
        return self.message[0..self.message_len];
    }

    pub fn isEmpty(self: *const Diagnostic) bool {
        return self.message_len == 0;
    }
};

threadlocal var current_diagnostic_ptr: ?*Diagnostic = null;

pub fn nativeError(err: anyerror, diag: ?*Diagnostic, comptime fmt: []const u8, args: anytype) anyerror {
    if (diag) |value| value.set(fmt, args);
    return err;
}

pub fn nativeErrorWithCurrent(err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    return nativeError(err, current_diagnostic_ptr, fmt, args);
}
