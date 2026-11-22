const std = @import("std");
const compute = @import("compute");

const allocator = std.heap.c_allocator;
const maximum_frame_bytes = 256 * 1024 * 1024;

pub const OutputSpec = struct {
    dtype: compute.DType,
    dimensions: []usize,

    pub fn deinit(self: *OutputSpec) void {
        allocator.free(self.dimensions);
    }

    pub fn tensorSpec(self: OutputSpec) !compute.TensorSpec {
        return compute.TensorSpec.init(allocator, self.dtype, self.dimensions, null);
    }

    pub fn byteCount(self: OutputSpec) !usize {
        var elements: usize = 1;
        for (self.dimensions) |dimension| elements = try std.math.mul(usize, elements, dimension);
        return std.math.mul(usize, elements, self.dtype.size());
    }
};

const WireOutput = struct {
    dtype: []const u8,
    shape: []const usize,
};

const WireCompileResponse = struct {
    ok: bool,
    input_count: usize = 0,
    outputs: []const WireOutput = &.{},
};

const WireEvidenceResponse = struct {
    ok: bool,
};

const WireErrorResponse = struct {
    ok: bool,
    code: ?[]const u8 = null,
};

const FailureCode = enum {
    generic,
    invalid_shape,
    invalid_command,
};

const Response = struct {
    storage: []u8,
    metadata: []const u8,
    payload: []u8,

    fn deinit(self: Response) void {
        allocator.free(self.storage);
    }
};

pub const Process = struct {
    threaded: std.Io.Threaded,
    child: std.process.Child,
    outputs: []OutputSpec = &.{},
    input_count: usize = 0,
    closed: bool = false,

    pub fn start() !*Process {
        const object = try allocator.create(Process);
        errdefer allocator.destroy(object);
        object.* = .{
            .threaded = std.Io.Threaded.init(allocator, .{}),
            .child = undefined,
        };
        errdefer object.threaded.deinit();
        const value = std.c.getenv("AFFON_CYNA_NATIVE");
        const path = if (value) |pointer| std.mem.span(pointer) else "cyna/affon-native/build/affon-cyna-native";
        object.child = try std.process.spawn(object.threaded.io(), .{
            .argv = &.{path},
            .stdin = .pipe,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        return object;
    }

    pub fn deinit(self: *Process) void {
        self.close();
        for (self.outputs) |*output| output.deinit();
        if (self.outputs.len != 0) allocator.free(self.outputs);
        self.threaded.deinit();
        allocator.destroy(self);
    }

    fn readExact(self: *Process, file: std.Io.File, output: []u8) !void {
        var offset: usize = 0;
        while (offset < output.len) {
            const count = try file.readStreaming(self.threaded.io(), &.{output[offset..]});
            if (count == 0) return error.EndOfStream;
            offset += count;
        }
    }

    fn receive(self: *Process) !Response {
        const stdout = self.child.stdout orelse return error.ProcessClosed;
        var header: [4]u8 = undefined;
        try self.readExact(stdout, &header);
        const frame_length = std.mem.readInt(u32, &header, .little);
        if (frame_length < 4 or frame_length > maximum_frame_bytes) return error.InvalidFrame;
        const storage = try allocator.alloc(u8, frame_length);
        errdefer allocator.free(storage);
        try self.readExact(stdout, storage);
        const metadata_length = std.mem.readInt(u32, storage[0..4], .little);
        if (metadata_length > storage.len - 4) return error.InvalidFrame;
        return .{
            .storage = storage,
            .metadata = storage[4 .. 4 + metadata_length],
            .payload = storage[4 + metadata_length ..],
        };
    }

    fn send(self: *Process, metadata: []const u8, payload: []const u8) !void {
        const stdin = self.child.stdin orelse return error.ProcessClosed;
        if (metadata.len > maximum_frame_bytes - 4) return error.FrameTooLarge;
        const frame_length = std.math.add(usize, 4 + metadata.len, payload.len) catch return error.FrameTooLarge;
        if (frame_length > maximum_frame_bytes) return error.FrameTooLarge;
        var outer: [4]u8 = undefined;
        std.mem.writeInt(u32, &outer, @intCast(frame_length), .little);
        var inner: [4]u8 = undefined;
        std.mem.writeInt(u32, &inner, @intCast(metadata.len), .little);
        try stdin.writeStreamingAll(self.threaded.io(), &outer);
        try stdin.writeStreamingAll(self.threaded.io(), &inner);
        try stdin.writeStreamingAll(self.threaded.io(), metadata);
        try stdin.writeStreamingAll(self.threaded.io(), payload);
    }

    fn failResponse(response: []const u8) ?FailureCode {
        const parsed = std.json.parseFromSlice(WireErrorResponse, allocator, response, .{
            .ignore_unknown_fields = true,
        }) catch return .generic;
        defer parsed.deinit();
        if (parsed.value.ok) return null;
        if (parsed.value.code) |code| {
            if (std.mem.eql(u8, code, "InvalidShape")) return .invalid_shape;
            if (std.mem.eql(u8, code, "InvalidCommand")) return .invalid_command;
        }
        return .generic;
    }

    pub fn compile(self: *Process, goal: []const u8, program_json: []const u8) !void {
        if (self.outputs.len != 0) return error.AlreadyCompiled;
        const metadata = try std.fmt.allocPrint(
            allocator,
            "{{\"command\":\"compile\",\"goal\":\"{s}\",\"program\":{s}}}",
            .{ goal, program_json },
        );
        defer allocator.free(metadata);
        try self.send(metadata, &.{});
        const response = try self.receive();
        defer response.deinit();
        var parsed = std.json.parseFromSlice(WireCompileResponse, allocator, response.metadata, .{
            .ignore_unknown_fields = true,
        }) catch return error.InvalidResponse;
        defer parsed.deinit();
        if (!parsed.value.ok) return error.CompileRejected;
        const outputs = try allocator.alloc(OutputSpec, parsed.value.outputs.len);
        errdefer allocator.free(outputs);
        var initialized: usize = 0;
        errdefer for (outputs[0..initialized]) |*output| output.deinit();
        for (parsed.value.outputs, 0..) |wire, index| {
            const dtype: compute.DType = if (std.mem.eql(u8, wire.dtype, "f32"))
                .f32
            else if (std.mem.eql(u8, wire.dtype, "f64"))
                .f64
            else if (std.mem.eql(u8, wire.dtype, "i64"))
                .i64
            else if (std.mem.eql(u8, wire.dtype, "u32"))
                .u32
            else if (std.mem.eql(u8, wire.dtype, "bf16"))
                .bf16
            else
                return error.InvalidDType;
            const dimensions = try allocator.dupe(usize, wire.shape);
            outputs[index] = .{ .dtype = dtype, .dimensions = dimensions };
            initialized += 1;
        }
        self.outputs = outputs;
        self.input_count = parsed.value.input_count;
    }

    pub fn run(self: *const Process, input_bytes: []const u8) ![]u8 {
        const mutable: *Process = @constCast(self);
        try mutable.send("{\"command\":\"run\"}", input_bytes);
        const response = try mutable.receive();
        if (failResponse(response.metadata)) |code| {
            response.deinit();
            return switch (code) {
                .generic => error.RunRejected,
                .invalid_shape => error.InvalidShape,
                .invalid_command => error.InvalidCommand,
            };
        }
        var expected: usize = 0;
        for (self.outputs) |output| expected = try std.math.add(usize, expected, try output.byteCount());
        if (response.payload.len != expected) {
            response.deinit();
            return error.InvalidOutputSize;
        }
        if (response.payload.len == 0) {
            response.deinit();
            return allocator.alloc(u8, 0);
        }
        return allocator.dupe(u8, response.payload);
    }

    pub fn evidence(self: *Process) ![]u8 {
        try self.send("{\"command\":\"evidence\"}", &.{});
        const response = try self.receive();
        defer response.deinit();
        if (failResponse(response.metadata)) |_| return error.EvidenceRejected;
        return allocator.dupe(u8, response.metadata);
    }

    pub fn close(self: *Process) void {
        if (self.closed) return;
        self.closed = true;
        const io = self.threaded.io();
        if (self.child.stdin) |stdin| {
            var sent = true;
            self.send("{\"command\":\"shutdown\"}", &.{}) catch {
                sent = false;
            };
            if (sent) {
                const response = self.receive() catch null;
                if (response) |value| value.deinit();
            }
            stdin.close(io);
            self.child.stdin = null;
        }
        std.process.Child.kill(&self.child, io);
        self.child = undefined;
    }
};
