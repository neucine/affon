const std = @import("std");

extern fn affon_metal_is_available() bool;
extern fn affon_metal_last_error() [*:0]const u8;
extern fn affon_metal_buffer_write(handle: *anyopaque, src: [*]const u8, byte_len: usize) c_int;
extern fn affon_metal_buffer_read(handle: *anyopaque, dst: [*]u8, byte_len: usize) c_int;

pub fn isAvailable() bool {
    return affon_metal_is_available();
}

pub fn lastError() []const u8 {
    return std.mem.span(affon_metal_last_error());
}

pub fn writeBuffer(handle: *anyopaque, src: []const u8) !void {
    if (affon_metal_buffer_write(handle, src.ptr, src.len) != 0) return error.MetalIoFailed;
}

pub fn readBuffer(handle: *anyopaque, dst: []u8) !void {
    if (affon_metal_buffer_read(handle, dst.ptr, dst.len) != 0) return error.MetalIoFailed;
}
