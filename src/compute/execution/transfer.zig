const std = @import("std");
const Value = @import("../types/tensor/value.zig").Value;
const kernel_dispatch = @import("../backend/dispatch.zig");

pub const TransferSummary = struct {
    to_host_count: usize = 0,
    to_host_bytes: usize = 0,
    from_host_count: usize = 0,
    from_host_bytes: usize = 0,
};

pub fn copyValueStorageInto(
    allocator: std.mem.Allocator,
    src: *const Value,
    dst: *Value,
) !TransferSummary {
    const byte_len = dst.shape.numel() * dst.dtype.size();
    const src_device = src.device() orelse return error.InputNotMaterialized;
    const dst_device = dst.device() orelse return error.InputNotMaterialized;
    if (src_device == dst_device) {
        try kernel_dispatch.copyStorage(
            src_device,
            try src.requireRuntimeBacking(),
            try dst.requireRuntimeBacking(),
            byte_len,
        );
        return .{};
    }
    if (src_device == .cpu and dst_device == .metal) {
        const src_bytes = try (try src.requireRuntimeBacking()).readableBytes();
        if (src_bytes.len != byte_len) return error.SizeMismatch;
        try (try dst.requireRuntimeBacking()).writeFromHost(src_bytes);
        return .{ .from_host_count = 1, .from_host_bytes = byte_len };
    }
    if (src_device == .metal and dst_device == .cpu) {
        const dst_bytes = try (try dst.requireRuntimeBacking()).writableBytes();
        if (dst_bytes.len != byte_len) return error.SizeMismatch;
        try (try src.requireRuntimeBacking()).copyToHost(dst_bytes);
        return .{ .to_host_count = 1, .to_host_bytes = byte_len };
    }
    const staging = try allocator.alignedAlloc(u8, .@"8", byte_len);
    defer allocator.free(staging);
    try (try src.requireRuntimeBacking()).copyToHost(staging);
    try (try dst.requireRuntimeBacking()).writeFromHost(staging);
    return .{
        .to_host_count = 1,
        .to_host_bytes = byte_len,
        .from_host_count = 1,
        .from_host_bytes = byte_len,
    };
}
