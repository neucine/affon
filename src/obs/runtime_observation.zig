const std = @import("std");
const builtin = @import("builtin");
const qjs = @import("../qjs.zig");
const memory_diag = @import("../mm/diagnostics.zig");

pub const VmDiagnostics = struct {
    phys_footprint_bytes: usize = 0,
    iokit_bytes: usize = 0,
    ioaccelerator_bytes: usize = 0,
    libc_malloc_in_use_bytes: usize = 0,
    libc_malloc_allocated_bytes: usize = 0,
};

pub fn refresh(rt: ?*qjs.c.JSRuntime) void {
    refreshVm();
    refreshQuickJs(rt);
}

pub fn refreshVm() void {
    const vm_stats = vmDiagnostics();
    memory_diag.observeVm(
        vm_stats.phys_footprint_bytes,
        vm_stats.iokit_bytes,
        vm_stats.ioaccelerator_bytes,
        vm_stats.libc_malloc_in_use_bytes,
        vm_stats.libc_malloc_allocated_bytes,
    );
}

pub fn refreshQuickJs(rt: ?*qjs.c.JSRuntime) void {
    if (rt) |runtime| memory_diag.observeQuickJs(qjs.computeMemoryUsage(runtime));
}

pub fn vmDiagnostics() VmDiagnostics {
    if (builtin.os.tag == .linux) return linuxVmDiagnostics();
    if (builtin.os.tag != .macos) return .{};
    return .{};
}

fn linuxVmDiagnostics() VmDiagnostics {
    var out: VmDiagnostics = .{};
    var file = std.fs.openFileAbsolute("/proc/self/statm", .{}) catch return out;
    defer file.close();

    var buf: [256]u8 = undefined;
    const len = file.readAll(&buf) catch return out;
    var fields = std.mem.tokenizeAny(u8, buf[0..len], " \t\r\n");
    _ = fields.next() orelse return out;
    const resident_pages_text = fields.next() orelse return out;
    const resident_pages = std.fmt.parseUnsigned(usize, resident_pages_text, 10) catch return out;
    out.phys_footprint_bytes = resident_pages * std.heap.pageSize();
    return out;
}
