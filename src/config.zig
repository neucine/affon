const std = @import("std");
const builtin = @import("builtin");
const compat = @import("support/compat.zig");
const mm = @import("mm/index.zig");
const c = @cImport({
    @cInclude("stdlib.h");
});

pub const Device = enum {
    cpu,
    metal,
};

pub const Config = struct {
    quickjs: QuickJS = .{},
    libuv: Libuv = .{},
    debug: Debug = .{},
    device: DeviceConfig = .{},
    csv: Csv = .{},
    repr: Repr = .{},
    observer: Observer = .{},

    pub const QuickJS = struct {
        stack_size: usize = 8 * 1024 * 1024,
    };
    pub const Libuv = struct {
        thread_pool_size: ?usize = null,
    };
    pub const Debug = struct {
        native_stack_trace: bool = false,
    };
    pub const DeviceConfig = struct {
        default: Device = .cpu,
        metal: Metal = .{},
        cpu: Cpu = .{},

        pub const Metal = struct {
            threadgroup_size: usize = 256,
            reduce_all_threshold: usize = 512,
            reduce_axis_threshold: usize = 512,
            pool_oversize_threshold_bytes: usize = 64 * 1024 * 1024,
        };
        pub const Cpu = struct {
            parallel_threshold: usize = 65536,
        };
    };
    pub const Csv = struct {
        chunk_size: usize = 65536,
    };
    pub const Repr = struct {
        max_items: usize = 6,
        repr_max_rows: usize = 20,
        repr_max_cols: usize = 12,
    };
    pub const Observer = struct {
        enabled: bool = false,
        port: usize = 0,
    };
};

pub var config: Config = .{};

fn parsePositiveUsize(s: []const u8) ?usize {
    const n = std.fmt.parseInt(usize, s, 10) catch return null;
    if (n == 0) return null;
    return n;
}

fn parseBool(s: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(s, "1")) return true;
    if (std.ascii.eqlIgnoreCase(s, "true")) return true;
    if (std.ascii.eqlIgnoreCase(s, "yes")) return true;
    if (std.ascii.eqlIgnoreCase(s, "on")) return true;
    if (std.ascii.eqlIgnoreCase(s, "0")) return false;
    if (std.ascii.eqlIgnoreCase(s, "false")) return false;
    if (std.ascii.eqlIgnoreCase(s, "no")) return false;
    if (std.ascii.eqlIgnoreCase(s, "off")) return false;
    return null;
}

fn parseDevice(s: []const u8) ?Device {
    if (std.ascii.eqlIgnoreCase(s, "cpu")) return .cpu;
    if (std.ascii.eqlIgnoreCase(s, "metal")) return .metal;
    return null;
}

fn loadUsize(key: []const u8, dest: *usize) void {
    const val = compat.getenv(key) orelse return;
    if (parsePositiveUsize(val)) |n| dest.* = n;
}

fn loadOptionalUsize(key: []const u8, dest: *?usize) void {
    const val = compat.getenv(key) orelse return;
    if (parsePositiveUsize(val)) |n| dest.* = n;
}

fn loadUsizeAllowZero(key: []const u8, dest: *usize) void {
    const val = compat.getenv(key) orelse return;
    const n = std.fmt.parseInt(usize, val, 10) catch return;
    dest.* = n;
}

fn loadDevice(key: []const u8, dest: *Device) void {
    const val = compat.getenv(key) orelse return;
    if (parseDevice(val)) |device| dest.* = device;
}

fn loadBool(key: []const u8, dest: *bool) void {
    const val = compat.getenv(key) orelse return;
    if (parseBool(val)) |flag| dest.* = flag;
}

fn setProcessEnv(key: [:0]const u8, value: []const u8) !void {
    if (builtin.os.tag == .windows) {
        return error.Unsupported;
    }

    const allocator = try mm.allocator(.runtime_host_scratch, .runtime_env_string);
    var value_z = try allocator.alloc(u8, value.len + 1);
    defer allocator.free(value_z);
    @memcpy(value_z[0..value.len], value);
    value_z[value.len] = 0;

    if (c.setenv(key, value_z.ptr, 1) != 0) {
        return error.SetEnvFailed;
    }
}

pub fn syncLibuvThreadPoolEnv() !void {
    const size = config.libuv.thread_pool_size orelse return;
    var buf: [32]u8 = undefined;
    const value = try std.fmt.bufPrint(&buf, "{d}", .{size});
    try setProcessEnv("UV_THREADPOOL_SIZE", value);
}

pub fn loadFromEnv() !void {
    loadDevice("AFFON_DEVICE", &config.device.default);
    loadUsize("AFFON_QJS_STACK_SIZE", &config.quickjs.stack_size);
    loadOptionalUsize("AFFON_LIBUV_THREADPOOL_SIZE", &config.libuv.thread_pool_size);
    loadBool("AFFON_NATIVE_STACK_TRACE", &config.debug.native_stack_trace);
    loadUsize("AFFON_METAL_THREADGROUP_SIZE", &config.device.metal.threadgroup_size);
    loadUsize("AFFON_METAL_REDUCE_ALL_THRESHOLD", &config.device.metal.reduce_all_threshold);
    loadUsize("AFFON_METAL_REDUCE_AXIS_THRESHOLD", &config.device.metal.reduce_axis_threshold);
    loadUsize("AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES", &config.device.metal.pool_oversize_threshold_bytes);
    loadUsize("AFFON_CPU_PARALLEL_THRESHOLD", &config.device.cpu.parallel_threshold);
    loadUsize("AFFON_CSV_CHUNK_SIZE", &config.csv.chunk_size);
    loadUsize("AFFON_REPR_MAX_ITEMS", &config.repr.max_items);
    loadUsize("AFFON_REPR_MAX_ROWS", &config.repr.repr_max_rows);
    loadUsize("AFFON_REPR_MAX_COLS", &config.repr.repr_max_cols);
    loadBool("AFFON_OBSERVER_ENABLED", &config.observer.enabled);
    loadUsizeAllowZero("AFFON_OBSERVER_PORT", &config.observer.port);
    // Legacy env aliases kept temporarily while the public config surface moves from ndarray.* to repr.*.
    loadUsize("AFFON_NDARRAY_FORMAT_MAX_ITEMS", &config.repr.max_items);
    loadUsize("AFFON_NDARRAY_REPR_MAX_ROWS", &config.repr.repr_max_rows);
    loadUsize("AFFON_NDARRAY_REPR_MAX_COLS", &config.repr.repr_max_cols);
    try syncLibuvThreadPoolEnv();
}

// C accessor for metal_bridge.m (ObjC cannot import Zig directly).
// Resolved at link time like all other affon_* symbols — no header needed.
export fn affon_config_metal_threadgroup_size() usize {
    return config.device.metal.threadgroup_size;
}

pub fn setDefaultDevice(device: Device) void {
    config.device.default = device;
}

pub fn getDefaultDevice() Device {
    return config.device.default;
}

test "Config defaults are correct" {
    const def = Config{};
    try std.testing.expectEqual(Device.cpu, def.device.default);
    try std.testing.expectEqual(@as(usize, 8 * 1024 * 1024), def.quickjs.stack_size);
    try std.testing.expectEqual(@as(?usize, null), def.libuv.thread_pool_size);
    try std.testing.expectEqual(false, def.debug.native_stack_trace);
    try std.testing.expectEqual(@as(usize, 256), def.device.metal.threadgroup_size);
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_all_threshold);
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_axis_threshold);
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), def.device.metal.pool_oversize_threshold_bytes);
    try std.testing.expectEqual(@as(usize, 65536), def.device.cpu.parallel_threshold);
    try std.testing.expectEqual(@as(usize, 65536), def.csv.chunk_size);
    try std.testing.expectEqual(@as(usize, 6), def.repr.max_items);
    try std.testing.expectEqual(@as(usize, 20), def.repr.repr_max_rows);
    try std.testing.expectEqual(@as(usize, 12), def.repr.repr_max_cols);
    try std.testing.expectEqual(false, def.observer.enabled);
    try std.testing.expectEqual(@as(usize, 0), def.observer.port);
}

test "loadUsize ignores missing env key" {
    var dest: usize = 42;
    loadUsize("AFFON_NONEXISTENT_KEY_ZZZYYYXXX", &dest);
    try std.testing.expectEqual(@as(usize, 42), dest);
}

test "loadOptionalUsize ignores missing env key" {
    var dest: ?usize = 42;
    loadOptionalUsize("AFFON_NONEXISTENT_OPTIONAL_KEY_ZZZYYYXXX", &dest);
    try std.testing.expectEqual(@as(?usize, 42), dest);
}

test "parsePositiveUsize rejects zero" {
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("0"));
}

test "parsePositiveUsize rejects non-numeric" {
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("notanumber"));
}

test "parsePositiveUsize accepts positive integer" {
    try std.testing.expectEqual(@as(?usize, 256), parsePositiveUsize("256"));
}

test "parseBool accepts common true and false values" {
    try std.testing.expectEqual(@as(?bool, true), parseBool("1"));
    try std.testing.expectEqual(@as(?bool, true), parseBool("true"));
    try std.testing.expectEqual(@as(?bool, true), parseBool("YES"));
    try std.testing.expectEqual(@as(?bool, false), parseBool("0"));
    try std.testing.expectEqual(@as(?bool, false), parseBool("false"));
    try std.testing.expectEqual(@as(?bool, false), parseBool("Off"));
    try std.testing.expectEqual(@as(?bool, null), parseBool("maybe"));
}

test "syncLibuvThreadPoolEnv mirrors config into UV_THREADPOOL_SIZE" {
    const old = compat.getenv("UV_THREADPOOL_SIZE");
    defer {
        if (old) |value| {
            _ = c.setenv("UV_THREADPOOL_SIZE", value.ptr, 1);
        } else {
            _ = c.unsetenv("UV_THREADPOOL_SIZE");
        }
        config.libuv.thread_pool_size = null;
    }

    config.libuv.thread_pool_size = 7;
    try syncLibuvThreadPoolEnv();
    try std.testing.expectEqualStrings("7", compat.getenv("UV_THREADPOOL_SIZE").?);
}

test "parseDevice accepts cpu and metal" {
    try std.testing.expectEqual(@as(?Device, .cpu), parseDevice("cpu"));
    try std.testing.expectEqual(@as(?Device, .metal), parseDevice("metal"));
}

test "parseDevice rejects unknown device" {
    try std.testing.expectEqual(@as(?Device, null), parseDevice("cuda"));
}
