const std = @import("std");
const zig_libs = @import("zig_libs");
const compute = @import("compute");

const cfg = zig_libs.config;
pub const Device = enum {
    cpu,
    metal,
    cuda,
};

pub const Config = struct {
    debug: Debug = .{},
    device: DeviceConfig = .{},
    csv: Csv = .{},
    repr: Repr = .{},

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
                .default = 512 * 1024 * 1024,
                .parser = .positive_int,
            }) = .{},
            pool_max_total_bytes: cfg.Startup(usize, .{
                .env = "AFFON_METAL_POOL_MAX_TOTAL_BYTES",
                .default = 1024 * 1024 * 1024,
                .parser = .positive_int,
            }) = .{},
            pool_size_classes: cfg.Startup([]const u8, .{
                .env = "AFFON_METAL_POOL_SIZE_CLASSES",
                .default = "65536:512,1048576:512,*:8",
                .parser = .non_empty_string,
            }) = .{},
        };
        pub const Cpu = struct {
            threads: cfg.Startup(usize, .{
                .env = "AFFON_CPU_THREADS",
                .default = 1,
                .parser = .positive_int,
            }) = .{},
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
        }) = .{},
        repr_max_rows: cfg.Runtime(usize, .{
            .env = "AFFON_REPR_MAX_ROWS",
            .default = 20,
            .parser = .positive_int,
        }) = .{},
        repr_max_cols: cfg.Runtime(usize, .{
            .env = "AFFON_REPR_MAX_COLS",
            .default = 12,
            .parser = .positive_int,
        }) = .{},
    };
};

pub var config = cfg.Store(Config).init();

pub fn loadFromEnv() !void {
    _ = try config.loadEnv();
    if (cfg.getenv("AFFON_DEVICE") == null) {
        const detected: Device = switch (@import("builtin").os.tag) {
            .macos => if (compute.backendAvailable(.metal)) .metal else .cpu,
            else => if (compute.backendAvailable(.cuda)) .cuda else .cpu,
        };
        config.set("device.default", detected) catch unreachable;
    }
    config.freezeStartup();
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

pub fn deviceName(device: Device) []const u8 {
    return switch (device) {
        .cpu => "cpu",
        .metal => "metal",
        .cuda => "cuda",
    };
}

pub fn deviceSelectionSource() []const u8 {
    return if (cfg.getenv("AFFON_DEVICE") != null) "AFFON_DEVICE" else "auto";
}

test "Config defaults are correct before runtime detection" {
    const def = Config{};
    try std.testing.expectEqual(Device.cpu, def.device.default.get());
    try std.testing.expectEqual(false, def.debug.native_stack_trace.get());
    try std.testing.expectEqual(@as(usize, 256), def.device.metal.threadgroup_size.get());
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_all_threshold.get());
    try std.testing.expectEqual(@as(usize, 512), def.device.metal.reduce_axis_threshold.get());
    try std.testing.expectEqual(@as(usize, 512 * 1024 * 1024), def.device.metal.pool_oversize_threshold_bytes.get());
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), def.device.metal.pool_max_total_bytes.get());
    try std.testing.expectEqualStrings("65536:512,1048576:512,*:8", def.device.metal.pool_size_classes.get());
    try std.testing.expectEqual(@as(usize, 65536), def.device.cpu.parallel_threshold.get());
    try std.testing.expectEqual(@as(usize, 65536), def.csv.chunk_size.get());
    try std.testing.expectEqual(@as(usize, 6), def.repr.max_items.get());
    try std.testing.expectEqual(@as(usize, 20), def.repr.repr_max_rows.get());
    try std.testing.expectEqual(@as(usize, 12), def.repr.repr_max_cols.get());
}

test "loadFromEnv reads schema-backed values" {
    const c = @cImport({
        @cInclude("stdlib.h");
    });
    const old_device = cfg.getenv("AFFON_DEVICE");
    const old_trace = cfg.getenv("AFFON_NATIVE_STACK_TRACE");
    defer {
        if (old_device) |value| {
            _ = c.setenv("AFFON_DEVICE", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_DEVICE");
        }
        if (old_trace) |value| {
            _ = c.setenv("AFFON_NATIVE_STACK_TRACE", value.ptr, 1);
        } else {
            _ = c.unsetenv("AFFON_NATIVE_STACK_TRACE");
        }
        config = cfg.Store(Config).init();
    }

    try cfg.setProcessEnv("AFFON_DEVICE", "metal");
    try cfg.setProcessEnv("AFFON_NATIVE_STACK_TRACE", "yes");

    config = cfg.Store(Config).init();
    _ = try config.loadEnv();

    try std.testing.expectEqual(Device.metal, config.read().device.default.get());
    try std.testing.expectEqual(true, config.read().debug.native_stack_trace.get());
}
