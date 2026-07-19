const qjs = @import("hao").qjs;

pub const ErrorCode = enum {
    invalid_arg,
    out_of_memory,
    internal,
};

pub fn jsError(ctx: ?*qjs.c.JSContext, code: ErrorCode, message: []const u8, _: anytype, _: anytype, _: anytype) qjs.c.JSValue {
    const global = qjs.c.JS_GetGlobalObject(ctx);
    defer qjs.freeValue(ctx, global);
    const ctor = qjs.getProperty(ctx, global, "AffonError");
    defer qjs.freeValue(ctx, ctor);
    if (!qjs.isFunction(ctx, ctor)) return qjs.c.JS_ThrowTypeError(ctx, "AffonError unavailable");
    var args = [_]qjs.c.JSValue{
        qjs.createString(ctx, @tagName(code)),
        qjs.createString(ctx, message),
    };
    defer qjs.freeValue(ctx, args[0]);
    defer qjs.freeValue(ctx, args[1]);
    const value = qjs.c.JS_CallConstructor(ctx, ctor, args.len, @ptrCast(&args));
    if (qjs.isException(value)) return value;
    return qjs.c.JS_Throw(ctx, value);
}

pub fn jsNativeError(ctx: ?*qjs.c.JSContext, code: ErrorCode, message: []const u8, _: anytype, _: anytype) qjs.c.JSValue {
    return jsError(ctx, code, message, null, null, null);
}

pub fn jsClassifiedNativeError(ctx: ?*qjs.c.JSContext, message: []const u8, _: anytype, _: anytype) qjs.c.JSValue {
    return jsError(ctx, .invalid_arg, message, null, null, null);
}

pub fn classifyNativeError(_: anytype) ErrorCode {
    return .internal;
}
