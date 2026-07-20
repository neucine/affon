const std = @import("std");
const affon = @import("affon");
const hao = @import("hao");

fn usage() void {
    std.debug.print(
        "Usage:\n  affon <file.ts|file.js>\n  affon run <file.ts|file.js>\n  affon test [--grep pattern] <file.ts|directory>...\n  affon --version\n",
        .{},
    );
}

fn runFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    var environment = try hao.RuntimeEnvironment.initWithIo(allocator, io, .{ .std = true });
    defer environment.deinit();
    try affon.register(&environment);
    environment.runFile(path) catch |err| {
        if (hao.module.lastError()) |message| std.debug.print("{s}\n", .{message});
        return err;
    };
    environment.runUntilIdle() catch |err| {
        if (hao.module.lastError()) |message| std.debug.print("{s}\n", .{message});
        return err;
    };
}

pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try hao.config.loadFromEnv();

    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    const command = args.next() orelse {
        usage();
        std.process.exit(2);
    };
    if (std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "version")) {
        if (args.next() != null) {
            usage();
            std.process.exit(2);
        }
        std.debug.print("affon {s}\n", .{affon.version});
        return;
    }
    if (std.mem.eql(u8, command, "run")) {
        const path = args.next() orelse {
            usage();
            std.process.exit(2);
        };
        if (args.next() != null) {
            usage();
            std.process.exit(2);
        }
        try runFile(allocator, init.io, path);
        return;
    }
    if (!std.mem.eql(u8, command, "test")) {
        if (args.next() != null) {
            usage();
            std.process.exit(2);
        }
        try runFile(allocator, init.io, command);
        return;
    }

    var grep: ?[]const u8 = null;
    var paths = std.ArrayList([]const u8).empty;
    defer paths.deinit(allocator);
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--grep")) {
            grep = args.next() orelse {
                usage();
                std.process.exit(2);
            };
        } else {
            try paths.append(allocator, arg);
        }
    }
    if (paths.items.len == 0) {
        usage();
        std.process.exit(2);
    }

    const result = try hao.test_runner.runWithPackageRegistrar(
        paths.items,
        grep,
        true,
        allocator,
        init.io,
        affon.registerPackage,
    );
    if (result.exitCode() != 0) std.process.exit(result.exitCode());
}
