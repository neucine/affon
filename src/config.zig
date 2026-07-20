const std = @import("std");
const zig_libs = @import("zig_libs");
const mm = @import("compute/memory.zig");

const cfg = zig_libs.config;
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
        stack_size: cfg.Startup(usize, .{
            .env = "AFFON_QJS_STACK_SIZE",
            .default = 8 * 1024 * 1024,
            .parser = .positive_int,
        }) = .{},
    };
    pub const Libuv = struct {
        thread_pool_size: cfg.Startup(?usize, .{
            .env = "AFFON_LIBUV_THREADPOOL_SIZE",
            .parser = .positive_int,
        }) = .{},
    };
    pub const Debug = struct {
        native_stack_trace: cfg.Runtime(bool, .{
            .env = "AFFON_NATIVE_STACK_TRACE",
        }) = .{},
    };
    pub const DeviceConfig = struct {
        default: cfg.Runtime(Device, .{
            .env = "AFFON_DEVICE",
            .default = .cpu,
            .parser = .enum_name,
        }) = .{},
        metal: Metal = .{},
        cpu: Cpu = .{},

        pub const Metal = struct {
            threadgroup_size: cfg.Startup(usize, .{
                .env = "AFFON_METAL_THREADGROUP_SIZE",
                .default = 256,
                .parser = .positive_int,
            }) = .{},
            reduce_all_threshold: cfg.Startup(usize, .{
                .env = "AFFON_METAL_REDUCE_ALL_THRESHOLD",
                .default = 512,
                .parser = .positive_int,
            }) = .{},
            reduce_axis_threshold: cfg.Startup(usize, .{
                .env = "AFFON_METAL_REDUCE_AXIS_THRESHOLD",
                .default = 512,
                .parser = .positive_int,
            }) = .{},
            pool_oversize_threshold_bytes: cfg.Startup(usize, .{
                .env = "AFFON_METAL_POOL_OVERSIZE_THRESHOLD_BYTES",
                .default = 64 * 1024 * 1024,
                .parser = .positive_int,
            }) = .{},
        };
        pub const Cpu = struct {
            parallel_threshold: cfg.Runtime(usize, .{
                .env = "AFFON_CPU_PARALLEL_THRESHOLD",
                .default = 65536,
                .parser = .positive_int,
            }) = .{},
        };
    };
    pub const Csv = struct {
        chunk_size: cfg.Runtime(usize, .{
            .env = "AFFON_CSV_CHUNK_SIZE",
            .default = 65536,
            .parser = .positive_int,
        }) = .{},
    };
    pub const Repr = struct {
        max_items: cfg.Runtime(usize, .{
            .env = "AFFON_REPR_MAX_ITEMS",
            .default = 6,
            .parser = .positive_int,
            .aliases = &.{cfg.deprecatedAlias("AFFON_NDARRAY_FORMAT_MAX_ITEMS")},
        }) = .{},
        repr_max_rows: cfg.Runtime(usize, .{
            .env = "AFFON_REPR_MAX_ROWS",
            .default = 20,
            .parser = .positive_int,
            .aliases = &.{cfg.deprecatedAlias("AFFON_NDARRAY_REPR_MAX_ROWS")},
        }) = .{},
        repr_max_cols: cfg.Runtime(usize, .{
            .env = "AFFON_REPR_MAX_COLS",
            .default = 12,
            .parser = .positive_int,
            .aliases = &.{cfg.deprecatedAlias("AFFON_NDARRAY_REPR_MAX_COLS")},
        }) = .{},
    };
    pub const Observer = struct {
        enabled: cfg.Runtime(bool, .{
            .env = "AFFON_OBSERVER_ENABLED",
        }) = .{},
        port: cfg.Runtime(usize, .{
            .env = "AFFON_OBSERVER_PORT",
            .parser = .int_allow_zero,
        }) = .{},
    };
};

pub var config = cfg.Store(Config).init();

pub fn syncLibuvThreadPoolEnv() !void {
    try cfg.syncOptionalUsizeEnv("UV_THREADPOOL_SIZE", config.read().libuv.thread_pool_size.get());
}

pub fn loadFromEnv() !void {
    _ = try config.loadEnv();
    config.freezeStartup();
    mm.setPoolOversizeThreshold(config.read().device.metal.pool_oversize_threshold_bytes.get());
    try syncLibuvThreadPoolEnv();
}

// C accessor for metal_bridge.m (ObjC cannot import Zig directly).
// Resolved at link time like all other affon_* symbols; no header needed.
export fn affon_config_metal_threadgroup_size() usize {
    return config.read().device.metal.threadgroup_size.get();
}

pub fn setDefaultDevice(device: Device) void {
    config.set("device.default", device) catch unreachable;
}

pub fn getDefaultDevice() Device {
    return config.read().device.default.get();
}

test "Config defaults are correct" {
    const def = Config{};
    try std.testing.expectEqual(Device.cpu, def.device.default.get());
    try std.testing.expectEqual(@as(usize, 8 * 1024 * 1024), def.quickjs.stack_size.get());
    try std.testing.expectEqual(@as(?usize, null), def.libuv.thread_pool_size.get());
    try std.testing.expectEqual(false, def.debug.native_stack_trace.get());
    try std.testing.expectEqual(@as(usize, 256), def.device.metal.threadgroup_size.get());
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_all_threshold.get());
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_axis_threshold.get());
    try std.testing.expectEqual(@as(usize, 64 * 1024 * 1024), def.device.metal.pool_oversize_threshold_bytes.get());
    try std.testing.expectEqual(@as(usize, 65536), def.device.cpu.parallel_threshold.get());
    try std.testing.expectEqual(@as(usize, 65536), def.csv.chunk_size.get());
    try std.testing.expectEqual(@as(usize, 6), def.repr.max_items.get());
    try std.testing.expectEqual(@as(usize, 20), def.repr.repr_max_rows.get());
    try std.testing.expectEqual(@as(usize, 12), def.repr.repr_max_cols.get());
    try std.testing.expectEqual(false, def.observer.enabled.get());
    try std.testing.expectEqual(@as(usize, 0), def.observer.port.get());
}

test "loadFromEnv reads schema-backed values" {
    const old_device = cfg.getenv("AFFON_DEVICE");
    const old_stack = cfg.getenv("AFFON_QJS_STACK_SIZE");
    const old_trace = cfg.getenv("AFFON_NATIVE_STACK_TRACE");
    defer {
        if (old_device) |value| {
            _ = c.setenv("AFFON_DEVICE", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_DEVICE");
        }
        if (old_stack) |value| {
            _ = c.setenv("AFFON_QJS_STACK_SIZE", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_QJS_STACK_SIZE");
        }
        if (old_trace) |value| {
            _ = c.setenv("AFFON_NATIVE_STACK_TRACE", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_NATIVE_STACK_TRACE");
        }
        config = cfg.Store(Config).init();
    }

    try cfg.setProcessEnv("AFFON_DEVICE", "metal");
    try cfg.setProcessEnv("AFFON_QJS_STACK_SIZE", "16");
    try cfg.setProcessEnv("AFFON_NATIVE_STACK_TRACE", "yes");

    config = cfg.Store(Config).init();
    _ = try config.loadEnv();

    try std.testing.expectEqual(Device.metal, config.read().device.default.get());
    try std.testing.expectEqual(@as(usize, 16), config.read().quickjs.stack_size.get());
    try std.testing.expectEqual(true, config.read().debug.native_stack_trace.get());
}

test "legacy repr aliases populate canonical fields" {
    const old_new = cfg.getenv("AFFON_REPR_MAX_ROWS");
    const old_legacy = cfg.getenv("AFFON_NDARRAY_REPR_MAX_ROWS");
    defer {
        if (old_new) |value| {
            _ = c.setenv("AFFON_REPR_MAX_ROWS", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_REPR_MAX_ROWS");
        }
        if (old_legacy) |value| {
            _ = c.setenv("AFFON_NDARRAY_REPR_MAX_ROWS", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_NDARRAY_REPR_MAX_ROWS");
        }
        config = cfg.Store(Config).init();
    }

    _ = c.unsetenv("AFFON_REPR_MAX_ROWS");
    try cfg.setProcessEnv("AFFON_NDARRAY_REPR_MAX_ROWS", "31");

    config = cfg.Store(Config).init();
    const report = try config.loadEnv();

    try std.testing.expectEqual(@as(usize, 31), config.read().repr.repr_max_rows.get());
    var saw_deprecated_alias = false;
    for (report.items()) |item| {
        if (std.mem.eql(u8, item.path, "repr.repr_max_rows")) saw_deprecated_alias = item.deprecated_alias;
    }
    try std.testing.expect(saw_deprecated_alias);
}

test "syncLibuvThreadPoolEnv mirrors config into UV_THREADPOOL_SIZE" {
    const old = cfg.getenv("UV_THREADPOOL_SIZE");
    defer {
        if (old) |value| {
            _ = c.setenv("UV_THREADPOOL_SIZE", value.ptr, 1);
        } else {
            _ = c.unsetenv("UV_THREADPOOL_SIZE");
        }
        config = cfg.Store(Config).init();
    }

    try config.set("libuv.thread_pool_size", @as(?usize, 7));
    try syncLibuvThreadPoolEnv();
    try std.testing.expectEqualStrings("7", cfg.getenv("UV_THREADPOOL_SIZE").?);
}
