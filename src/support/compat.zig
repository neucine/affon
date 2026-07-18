const std = @import("std");

var runtime_io: ?std.Io = null;

pub fn setRuntimeIo(io: std.Io) void {
    runtime_io = io;
}

pub fn runtimeIo() ?std.Io {
    return runtime_io;
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
    if (@hasDecl(std.fs, "cwd")) {
        return std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
    }

    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);

    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.ReadFailed;
    if (@as(u64, @intCast(end)) > max_bytes) return error.FileTooBig;
    if (std.c.lseek(fd, 0, std.c.SEEK.SET) < 0) return error.ReadFailed;

    const buf = try allocator.alloc(u8, @intCast(end));
    errdefer allocator.free(buf);

    var offset: usize = 0;
    while (offset < buf.len) {
        const n = std.c.read(fd, buf[offset..].ptr, buf.len - offset);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        offset += @intCast(n);
    }
    return buf[0..offset];
}

pub const FdFile = struct {
    fd: std.c.fd_t,

    pub fn openRead(path: []const u8) !FdFile {
        if (@hasDecl(std.fs, "cwd")) {
            @compileError("FdFile is intended for Zig std.fs-free builds");
        }
        const path_z = try std.heap.page_allocator.dupeZ(u8, path);
        defer std.heap.page_allocator.free(path_z);
        const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.FileNotFound;
        return .{ .fd = fd };
    }

    pub fn create(path: []const u8) !FdFile {
        if (@hasDecl(std.fs, "cwd")) {
            @compileError("FdFile is intended for Zig std.fs-free builds");
        }
        const path_z = try std.heap.page_allocator.dupeZ(u8, path);
        defer std.heap.page_allocator.free(path_z);
        const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o666));
        if (fd < 0) return error.OpenFailed;
        return .{ .fd = fd };
    }

    pub fn close(self: FdFile) void {
        _ = std.c.close(self.fd);
    }

    pub fn size(self: FdFile) !u64 {
        const current = std.c.lseek(self.fd, 0, std.c.SEEK.CUR);
        if (current < 0) return error.SeekFailed;
        const end = std.c.lseek(self.fd, 0, std.c.SEEK.END);
        if (end < 0) return error.SeekFailed;
        if (std.c.lseek(self.fd, current, std.c.SEEK.SET) < 0) return error.SeekFailed;
        return @intCast(end);
    }

    pub fn read(self: FdFile, buf: []u8) !usize {
        const n = std.c.read(self.fd, buf.ptr, buf.len);
        if (n < 0) return error.ReadFailed;
        return @intCast(n);
    }

    pub fn seekTo(self: FdFile, pos: u64) !void {
        if (std.c.lseek(self.fd, @intCast(pos), std.c.SEEK.SET) < 0) return error.SeekFailed;
    }

    pub fn writeAll(self: FdFile, bytes: []const u8) !void {
        var remaining = bytes;
        while (remaining.len > 0) {
            const written = std.c.write(self.fd, remaining.ptr, remaining.len);
            if (written <= 0) return error.WriteFailed;
            remaining = remaining[@intCast(written)..];
        }
    }
};

pub fn writeFile(path: []const u8, bytes: []const u8) !void {
    if (@hasDecl(std.fs, "cwd")) {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        return file.writeAll(bytes);
    }
    const file = try FdFile.create(path);
    defer file.close();
    try file.writeAll(bytes);
}

pub fn makePath(path: []const u8) !void {
    if (@hasDecl(std.fs, "cwd")) {
        return std.fs.cwd().makePath(path);
    }

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return error.NameTooLong;
    @memcpy(buf[0..path.len], path);

    var i: usize = if (path.len > 0 and path[0] == '/') 1 else 0;
    while (i <= path.len) : (i += 1) {
        if (i != path.len and buf[i] != '/') continue;
        if (i == 0) continue;
        const save = if (i < path.len) buf[i] else 0;
        if (i < path.len) buf[i] = 0 else buf[i] = 0;
        const part: [*:0]u8 = @ptrCast(&buf);
        if (std.c.mkdir(part, 0o755) != 0 and !pathExists(buf[0..i])) return error.MakePathFailed;
        if (i < path.len) buf[i] = save;
        if (i == path.len) break;
    }
}

pub fn realpathAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (@hasDecl(std.fs, "realpathAlloc")) {
        return std.fs.realpathAlloc(allocator, path);
    }

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    const resolved = std.c.realpath(path_z.ptr, &buf) orelse return error.FileNotFound;
    return allocator.dupe(u8, std.mem.span(resolved));
}

pub fn selfExeDirPathAlloc(allocator: std.mem.Allocator) ![]u8 {
    if (@hasDecl(std.fs, "selfExeDirPathAlloc")) {
        return std.fs.selfExeDirPathAlloc(allocator);
    }

    const exe_path = try selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    return allocator.dupe(u8, std.fs.path.dirname(exe_path) orelse ".");
}

pub fn selfExePathAlloc(allocator: std.mem.Allocator) ![]u8 {
    if (@hasDecl(std.fs, "selfExePathAlloc")) {
        return std.fs.selfExePathAlloc(allocator);
    }
    if (@hasDecl(std.c, "_NSGetExecutablePath")) {
        var size: u32 = 0;
        _ = std.c._NSGetExecutablePath(undefined, &size);
        const buf = try allocator.alloc(u8, size);
        defer allocator.free(buf);
        if (std.c._NSGetExecutablePath(buf.ptr, &size) != 0) return error.NameTooLong;
        return allocator.dupe(u8, std.mem.sliceTo(buf, 0));
    }
    return error.Unsupported;
}

pub fn pathExists(path: []const u8) bool {
    if (@hasDecl(std.fs, "cwd")) {
        std.fs.cwd().access(path, .{}) catch return false;
        return true;
    }
    const path_z = std.heap.page_allocator.dupeZ(u8, path) catch return false;
    defer std.heap.page_allocator.free(path_z);
    return std.c.access(path_z.ptr, std.c.F_OK) == 0;
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
