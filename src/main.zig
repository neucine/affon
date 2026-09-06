const std = @import("std");
const builtin = @import("builtin");
const affon = @import("affon");
const hao = @import("hao");

fn usage() void {
    std.debug.print(
        "Usage:\n  affon <file.ts|file.js>\n  affon run <file.ts|file.js>\n  affon test [--grep pattern] <file.ts|directory>...\n  affon jupyter --connection-file <file>\n  affon jupyter install\n  affon --version\n",
        .{},
    );
}

fn printStartupInfo() void {
    std.debug.print(
        "affon {s} · device={s} · selection={s} · platform={s}-{s}\n",
        .{
            affon.version,
            affon.config.deviceName(affon.config.getDefaultDevice()),
            affon.config.deviceSelectionSource(),
            @tagName(builtin.os.tag),
            @tagName(builtin.cpu.arch),
        },
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

    try affon.memory.init(allocator);
    defer {
        if (affon.memory.deinit() == .leak) {
            var stderr_buffer: [1024]u8 = undefined;
            const stderr = std.debug.lockStderr(&stderr_buffer);
            defer std.debug.unlockStderr();
            stderr.file_writer.interface.writeAll("Affon memory leaks:\n") catch {};
            affon.memory.writeLeakReport(&stderr.file_writer.interface) catch {};
        }
    }
    hao.runtime_allocator.init(affon.memory.allocator(.hao_runtime));
    try hao.config.loadFromEnv();
    try affon.config.loadFromEnv();
    try affon.compute.memory.init(allocator);
    defer {
        if (affon.compute.memory.deinit() == .leak) {
            var stderr_buffer: [1024]u8 = undefined;
            const stderr = std.debug.lockStderr(&stderr_buffer);
            defer std.debug.unlockStderr();
            stderr.file_writer.interface.writeAll("compute memory leaks:\n") catch {};
            affon.compute.memory.writeLeakReport(&stderr.file_writer.interface) catch {};
        }
    }

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
    printStartupInfo();
    if (std.mem.eql(u8, command, "run")) {
        const path = args.next() orelse {
            usage();
            std.process.exit(2);
        };
        if (args.next() != null) {
            usage();
            std.process.exit(2);
        }
        try runFile(affon.memory.allocator(.hao_runtime), init.io, path);
        return;
    }
    if (std.mem.eql(u8, command, "jupyter")) {
        const subcommand = args.next() orelse {
            usage();
            std.process.exit(2);
        };
        if (std.mem.eql(u8, subcommand, "install")) {
            if (args.next() != null) {
                usage();
                std.process.exit(2);
            }
            try hao.jupyter.install(init.io);
            return;
        }
        if (std.mem.eql(u8, subcommand, "--connection-file")) {
            const connection_file = args.next() orelse {
                usage();
                std.process.exit(2);
            };
            if (args.next() != null) {
                usage();
                std.process.exit(2);
            }
            try hao.jupyter.runWithPackageRegistrar(connection_file, init.io, affon.registerPackage);
            return;
        }
        usage();
        std.process.exit(2);
    }
    if (!std.mem.eql(u8, command, "test")) {
        if (args.next() != null) {
            usage();
            std.process.exit(2);
        }
        try runFile(affon.memory.allocator(.hao_runtime), init.io, command);
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
        affon.memory.allocator(.hao_runtime),
        init.io,
        affon.registerPackage,
    );
    if (result.exitCode() != 0) std.process.exit(result.exitCode());
}
