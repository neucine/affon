const std = @import("std");
const builtin = @import("builtin");

const FileTime = extern struct { low: u32, high: u32 };
extern "kernel32" fn GetSystemTimeAsFileTime(*FileTime) void;

var runtime_io: ?std.Io = null;

pub fn setRuntimeIo(io: std.Io) void {
    runtime_io = io;
}

pub fn runtimeIo() ?std.Io {
    return runtime_io;
}

fn requireRuntimeIo() !std.Io {
    return runtime_io orelse error.IoUnavailable;
}

pub const Mutex = struct {
    locked: std.atomic.Value(bool) = .init(false),

    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) {
            while (self.locked.load(.monotonic)) {
                std.atomic.spinLoopHint();
            }
        }
    }

    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

pub fn getenv(key: []const u8) ?[]const u8 {
    if (@hasDecl(std.posix, "getenv")) {
        return std.posix.getenv(key);
    }

    var key_z: [256:0]u8 = undefined;
    if (key.len >= key_z.len) return null;
    @memcpy(key_z[0..key.len], key);
    key_z[key.len] = 0;

    const value = std.c.getenv(&key_z) orelse return null;
    return std.mem.span(value);
}

pub fn nanoTimestamp() i128 {
    if (builtin.os.tag == .windows) {
        var timestamp: FileTime = undefined;
        GetSystemTimeAsFileTime(&timestamp);
        const ticks: u64 = (@as(u64, timestamp.high) << 32) | timestamp.low;
        return (@as(i128, ticks) - 116444736000000000) * 100;
    }
    if (@hasDecl(std.time, "nanoTimestamp")) {
        return std.time.nanoTimestamp();
    }

    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.REALTIME, &ts) != 0) return 0;
    return (@as(i128, ts.sec) * std.time.ns_per_s) + @as(i128, ts.nsec);
}

pub fn milliTimestamp() i64 {
    if (@hasDecl(std.time, "milliTimestamp")) {
        return std.time.milliTimestamp();
    }
    return @intCast(@divFloor(nanoTimestamp(), std.time.ns_per_ms));
}

pub fn trimRight(comptime T: type, slice: []const T, values_to_strip: []const T) []const T {
    if (@hasDecl(std.mem, "trimRight")) {
        return std.mem.trimRight(T, slice, values_to_strip);
    }

    var end = slice.len;
    while (end > 0 and std.mem.indexOfScalar(T, values_to_strip, slice[end - 1]) != null) {
        end -= 1;
    }
    return slice[0..end];
}

pub fn trimLeft(comptime T: type, slice: []const T, values_to_strip: []const T) []const T {
    if (@hasDecl(std.mem, "trimLeft")) {
        return std.mem.trimLeft(T, slice, values_to_strip);
    }

    var start: usize = 0;
    while (start < slice.len and std.mem.indexOfScalar(T, values_to_strip, slice[start]) != null) {
        start += 1;
    }
    return slice[start..];
}

pub fn writeStdout(bytes: []const u8) void {
    writeFd(std.c.STDOUT_FILENO, bytes);
}

pub fn writeStderr(bytes: []const u8) void {
    writeFd(std.c.STDERR_FILENO, bytes);
}

pub fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(try requireRuntimeIo(), path, allocator, .limited(max_bytes));
}

pub const FdFile = struct {
    file: std.Io.File,
    io: std.Io,

    pub fn openRead(path: []const u8) !FdFile {
        const io = try requireRuntimeIo();
        return .{ .file = try std.Io.Dir.cwd().openFile(io, path, .{}), .io = io };
    }

    pub fn create(path: []const u8) !FdFile {
        const io = try requireRuntimeIo();
        return .{ .file = try std.Io.Dir.cwd().createFile(io, path, .{}), .io = io };
    }

    pub fn close(self: FdFile) void {
        self.file.close(self.io);
    }

    pub fn size(self: FdFile) !u64 {
        return (try self.file.stat(self.io)).size;
    }

    pub fn read(self: FdFile, buf: []u8) !usize {
        return self.file.readStreaming(self.io, &.{buf}) catch |err| switch (err) {
            error.EndOfStream => 0,
            else => return err,
        };
    }

    pub fn seekTo(self: FdFile, pos: u64) !void {
        try self.io.vtable.fileSeekTo(self.io.userdata, self.file, pos);
    }

    pub fn writeAll(self: FdFile, bytes: []const u8) !void {
        return self.file.writeStreamingAll(self.io, bytes);
    }
};

pub fn writeFile(path: []const u8, bytes: []const u8) !void {
    const io = try requireRuntimeIo();
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    return file.writeStreamingAll(io, bytes);
}

pub fn makePath(path: []const u8) !void {
    return std.Io.Dir.cwd().createDirPath(try requireRuntimeIo(), path);
}

pub fn realpathAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().realPathFileAlloc(try requireRuntimeIo(), path, allocator);
}

pub fn selfExeDirPathAlloc(allocator: std.mem.Allocator) ![]u8 {
    const exe_path = try selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    return allocator.dupe(u8, std.fs.path.dirname(exe_path) orelse ".");
}

pub fn selfExePathAlloc(allocator: std.mem.Allocator) ![]u8 {
    return std.process.executablePathAlloc(try requireRuntimeIo(), allocator);
}

pub fn pathExists(path: []const u8) bool {
    std.Io.Dir.cwd().access(runtime_io orelse return false, path, .{}) catch return false;
    return true;
}

pub fn randomBytes(bytes: []u8) void {
    if (@hasDecl(std.crypto, "random")) {
        std.crypto.random.bytes(bytes);
        return;
    }
    std.c.arc4random_buf(bytes.ptr, bytes.len);
}

fn writeFd(fd: std.c.fd_t, bytes: []const u8) void {
    var remaining = bytes;
    while (remaining.len > 0) {
        const written = std.c.write(fd, remaining.ptr, remaining.len);
        if (written <= 0) return;
        remaining = remaining[@intCast(written)..];
    }
}
