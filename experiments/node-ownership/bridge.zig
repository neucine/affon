const std = @import("std");
const compute = @import("compute");
const allocator = std.heap.page_allocator;

const Handle = struct { session: *compute.Session, children: usize = 0, closed: bool = false, host_alive: bool = true, released: bool = false };
const Value = struct { owner: *Handle, tensor: ?*compute.Tensor };

export fn affon_session_new() ?*Handle {
    const session = compute.createSession(allocator, .cpu) catch return null;
    const handle = allocator.create(Handle) catch { session.deinit() catch {}; return null; };
    handle.* = .{ .session = session };
    return handle;
}

fn finish(handle: *Handle) void {
    if (!handle.closed or handle.children != 0) return;
    if (!handle.released) {
        handle.session.deinit() catch unreachable;
        handle.released = true;
    }
    if (!handle.host_alive) allocator.destroy(handle);
}

export fn affon_session_close(handle: *Handle) void {
    if (handle.closed) return;
    handle.closed = true;
    finish(handle);
}

export fn affon_session_destroy(handle: *Handle) void {
    affon_session_close(handle);
    handle.host_alive = false;
    finish(handle);
}

export fn affon_tensor_new(handle: *Handle, count: usize) ?*Value {
    if (handle.closed or count == 0 or count > 16 * 1024 * 1024) return null;
    var spec = compute.TensorSpec.init(allocator, .f32, &.{count}, null) catch return null;
    defer spec.deinit();
    const tensor = handle.session.createOutput(spec) catch return null;
    const value = allocator.create(Value) catch { tensor.deinit(); return null; };
    value.* = .{ .owner = handle, .tensor = tensor };
    handle.children += 1;
    return value;
}

export fn affon_tensor_close(value: *Value) void {
    if (value.tensor) |tensor| {
        tensor.deinit();
        value.tensor = null;
        const handle = value.owner;
        handle.children -= 1;
        finish(handle);
    }
}

export fn affon_tensor_destroy(value: *Value) void {
    affon_tensor_close(value);
    allocator.destroy(value);
}

export fn affon_active_bytes(handle: *Handle) usize {
    return if (handle.released) 0 else handle.session.memoryStats().active_bytes;
}

export fn affon_live_tensors(handle: *Handle) usize { return handle.children; }
