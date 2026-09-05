const std = @import("std");
const errors = @import("errors.zig");
const qjs = @import("hao").qjs;
const js_classes = @import("js_classes.zig");
const cfg = @import("../../config.zig");
const compute = @import("compute");
const tensor_binding = @import("../compute/native.zig");
const DataFrame = @import("DataFrame.zig").DataFrame;
const pipeline = @import("pipeline.zig");
const csv = @import("csv.zig");

const DType = compute.DType;

const data_alloc = std.heap.c_allocator;
fn bridgeStringAlloc() std.mem.Allocator {
    return std.heap.c_allocator;
}
fn bridgeStateAlloc() std.mem.Allocator {
    return std.heap.c_allocator;
}

pub const specifier: [:0]const u8 = "affon:dataset/native";

fn parseToTensorDType(ctx: ?*qjs.c.JSContext, argc: c_int, argv: [*c]qjs.c.JSValueConst) !DType {
    if (argc < 1 or !qjs.isObject(argv[0])) return .f32;
    const dtype_value = qjs.getProperty(ctx, argv[0], "dtype");
    defer qjs.freeValue(ctx, dtype_value);
    if (qjs.isUndefined(dtype_value) or qjs.isNull(dtype_value)) return .f32;
    if (!qjs.isString(dtype_value)) return error.InvalidArgument;

    const text = qjs.valueToStringAlloc(ctx, dtype_value, data_alloc) catch return error.OutOfMemory;
    defer data_alloc.free(text);

    if (std.mem.eql(u8, text, "f32")) return .f32;
    if (std.mem.eql(u8, text, "f64")) return .f64;
    return error.InvalidArgument;
}

pub fn load(ctx: ?*anyopaque, module_name: [*c]const u8) ?*anyopaque {
    const qjs_ctx: ?*qjs.c.JSContext = @ptrCast(ctx);
    const module = qjs.c.JS_NewCModule(qjs_ctx, module_name, init) orelse return null;
    _ = qjs.c.JS_AddModuleExport(qjs_ctx, module, "readNative");
    return module;
}

fn init(ctx: ?*qjs.c.JSContext, module: ?*qjs.c.JSModuleDef) callconv(.c) c_int {
    initClass(ctx) catch return -1;
    const read_fn = qjs.c.JS_NewCFunction(ctx, js_readNative, "readNative", 1);
    if (qjs.isException(read_fn)) return -1;
    return qjs.c.JS_SetModuleExport(ctx, module, "readNative", read_fn);
}

const SharedDf = struct {
    df: *DataFrame,
    ref_count: u32,

    fn create(df: *DataFrame) !*SharedDf {
        const s = try bridgeStateAlloc().create(SharedDf);
        s.* = .{ .df = df, .ref_count = 1 };
        return s;
    }

    fn retain(self: *SharedDf) void {
        self.ref_count += 1;
    }

    fn release(self: *SharedDf) void {
        self.ref_count -= 1;
        if (self.ref_count == 0) {
            self.df.deinit();
            bridgeStateAlloc().destroy(self);
        }
    }
};

const Dataset = struct {
    shared: *SharedDf,
    steps: std.ArrayList(pipeline.Step),

    fn init(df: *DataFrame) !*Dataset {
        const shared = try SharedDf.create(df);
        errdefer shared.release();
        const ds = try bridgeStateAlloc().create(Dataset);
        ds.* = .{
            .shared = shared,
            .steps = .empty,
        };
        return ds;
    }

    fn deinit(self: *Dataset) void {
        for (self.steps.items) |step| freeStep(step);
        self.steps.deinit(bridgeStateAlloc());
        self.shared.release();
        bridgeStateAlloc().destroy(self);
    }

    fn addStep(self: *const Dataset, step: pipeline.Step) !*Dataset {
        const ds = try bridgeStateAlloc().create(Dataset);
        errdefer bridgeStateAlloc().destroy(ds);
        ds.* = .{
            .shared = self.shared,
            .steps = .empty,
        };
        self.shared.retain();
        for (self.steps.items) |s| {
            try ds.steps.append(bridgeStateAlloc(), try dupeStep(s));
        }
        try ds.steps.append(bridgeStateAlloc(), step);
        return ds;
    }
};

fn freeStep(step: pipeline.Step) void {
    switch (step) {
        .select => |cols| freeStringSlice(cols),
        .drop => |cols| freeStringSlice(cols),
        .encode_label => |s| bridgeStringAlloc().free(s),
        .encode_onehot => |s| bridgeStringAlloc().free(s),
        .normalize => |cols| freeStringSlice(cols),
        .standardize => |cols| freeStringSlice(cols),
        .log_transform => |cols| freeStringSlice(cols),
        .clip => |op| freeStringSlice(op.columns),
        .fillna => |op| bridgeStringAlloc().free(op.column),
        .dropna => |cols| freeStringSlice(cols),
        .rename => |op| {
            bridgeStringAlloc().free(op.old_name);
            bridgeStringAlloc().free(op.new_name);
        },
        .features => |cols| freeStringSlice(cols),
        .target => |s| bridgeStringAlloc().free(s),
        .shuffle, .sample => {},
    }
}

fn dupeStep(step: pipeline.Step) !pipeline.Step {
    return switch (step) {
        .select => |cols| .{ .select = try dupeStringSlice(cols) },
        .drop => |cols| .{ .drop = try dupeStringSlice(cols) },
        .encode_label => |s| .{ .encode_label = try bridgeStringAlloc().dupe(u8, s) },
        .encode_onehot => |s| .{ .encode_onehot = try bridgeStringAlloc().dupe(u8, s) },
        .normalize => |cols| .{ .normalize = try dupeStringSlice(cols) },
        .standardize => |cols| .{ .standardize = try dupeStringSlice(cols) },
        .log_transform => |cols| .{ .log_transform = try dupeStringSlice(cols) },
        .clip => |op| .{ .clip = .{ .columns = try dupeStringSlice(op.columns), .min_val = op.min_val, .max_val = op.max_val } },
        .fillna => |op| .{ .fillna = .{ .column = try bridgeStringAlloc().dupe(u8, op.column), .value = op.value } },
        .dropna => |cols| .{ .dropna = try dupeStringSlice(cols) },
        .rename => |op| .{ .rename = .{ .old_name = try bridgeStringAlloc().dupe(u8, op.old_name), .new_name = try bridgeStringAlloc().dupe(u8, op.new_name) } },
        .features => |cols| .{ .features = try dupeStringSlice(cols) },
        .target => |s| .{ .target = try bridgeStringAlloc().dupe(u8, s) },
        .shuffle => .{ .shuffle = {} },
        .sample => |n| .{ .sample = n },
    };
}

fn dupeStringSlice(src: []const []const u8) ![][]const u8 {
    const result = try bridgeStringAlloc().alloc([]const u8, src.len);
    for (src, 0..) |s, i| result[i] = try bridgeStringAlloc().dupe(u8, s);
    return result;
}

fn freeStringSlice(slice: []const []const u8) void {
    for (slice) |s| bridgeStringAlloc().free(s);
    bridgeStringAlloc().free(slice);
}

fn initClass(ctx: ?*qjs.c.JSContext) !void {
    const runtime = qjs.c.JS_GetRuntime(ctx);
    const class_id = js_classes.ensureDatasetClassId(runtime);
    var class_def = std.mem.zeroInit(qjs.c.JSClassDef, .{
        .class_name = "Dataset",
        .finalizer = js_finalize,
    });
    _ = qjs.c.JS_NewClass(runtime, class_id, &class_def);
}

fn js_finalize(_: ?*qjs.c.JSRuntime, value: qjs.c.JSValue) callconv(.c) void {
    const class_id = js_classes.ensureDatasetClassId(null);
    const raw = qjs.c.JS_GetOpaque(value, class_id);
    if (raw == null) return;
    const ds: *Dataset = @ptrCast(@alignCast(raw));
    ds.deinit();
}

fn createDatasetObject(ctx: ?*qjs.c.JSContext, ds: *Dataset) qjs.c.JSValue {
    const class_id = js_classes.ensureDatasetClassId(qjs.c.JS_GetRuntime(ctx));
    const obj = qjs.c.JS_NewObjectClass(ctx, class_id);
    if (qjs.isException(obj)) return obj;
    _ = qjs.c.JS_SetOpaque(obj, ds);

    const cols = makeStringArray(ctx, ds.shared.df.col_names) orelse return qjs.exceptionValue();
    if (qjs.c.JS_DefinePropertyValueStr(ctx, obj, "columns", cols, qjs.c.JS_PROP_HAS_VALUE | qjs.c.JS_PROP_HAS_ENUMERABLE | qjs.c.JS_PROP_ENUMERABLE) < 0) {
        return qjs.exceptionValue();
    }

    inline for ([_]struct {
        name: [:0]const u8,
        func: *const fn (?*qjs.c.JSContext, qjs.c.JSValueConst, c_int, [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue,
        len: c_int,
    }{
        .{ .name = "select", .func = js_select, .len = 1 },
        .{ .name = "drop", .func = js_drop, .len = 1 },
        .{ .name = "encode", .func = js_encode, .len = 1 },
        .{ .name = "normalize", .func = js_normalize, .len = 1 },
        .{ .name = "standardize", .func = js_standardize, .len = 1 },
        .{ .name = "log", .func = js_log, .len = 1 },
        .{ .name = "clip", .func = js_clip, .len = 3 },
        .{ .name = "shuffle", .func = js_shuffle, .len = 0 },
        .{ .name = "sample", .func = js_sample, .len = 1 },
        .{ .name = "fillna", .func = js_fillna, .len = 2 },
        .{ .name = "dropna", .func = js_dropna, .len = 1 },
        .{ .name = "rename", .func = js_rename, .len = 2 },
        .{ .name = "features", .func = js_features, .len = 1 },
        .{ .name = "target", .func = js_target, .len = 1 },
        .{ .name = "concat", .func = js_concat, .len = 1 },
        .{ .name = "split", .func = js_split, .len = 1 },
        .{ .name = "toTensor", .func = js_toTensor, .len = 1 },
        .{ .name = "toTensors", .func = js_toTensors, .len = 1 },
    }) |entry| {
        const fn_value = qjs.c.JS_NewCFunction(ctx, entry.func, entry.name.ptr, entry.len);
        if (qjs.isException(fn_value)) return qjs.exceptionValue();
        if (qjs.c.JS_SetPropertyStr(ctx, obj, entry.name.ptr, fn_value) < 0) return qjs.exceptionValue();
    }
    return obj;
}

fn extractDataset(ctx: ?*qjs.c.JSContext, value: qjs.c.JSValueConst) ?*Dataset {
    const class_id = js_classes.ensureDatasetClassId(qjs.c.JS_GetRuntime(ctx));
    const raw = qjs.c.JS_GetOpaque2(ctx, value, class_id) orelse return null;
    return @ptrCast(@alignCast(raw));
}

fn js_readNative(ctx: ?*qjs.c.JSContext, _: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "read() requires a path argument", null, null, null);
    const path = qjs.valueToStringAlloc(ctx, argv[0], bridgeStringAlloc()) catch return errors.jsError(ctx, .invalid_arg, "read() requires a string path", null, null, null);
    defer bridgeStringAlloc().free(path);

    const df = csv.parseCSV(data_alloc, path) catch |err| {
        var buf: [256]u8 = undefined;
        const msg = std.fmt.bufPrintZ(&buf, "read(): {s}", .{@errorName(err)}) catch "read(): parse error";
        return errors.jsClassifiedNativeError(ctx, msg, err, null);
    };
    const ds = Dataset.init(df) catch {
        df.deinit();
        return errors.jsError(ctx, .out_of_memory, "read(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, ds);
}

fn js_select(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .select);
}

fn js_drop(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .drop);
}

fn js_normalize(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .normalize);
}

fn js_standardize(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .standardize);
}

fn js_log(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .log_transform);
}

fn js_features(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    return pipelineStringArgs(ctx, this, argc, argv, .features);
}

fn js_target(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "target() requires a column name", null, null, null);
    const col_name = qjs.valueToStringAlloc(ctx, argv[0], bridgeStringAlloc()) catch return errors.jsError(ctx, .invalid_arg, "target() requires a string column name", null, null, null);
    const new_ds = ds.addStep(.{ .target = col_name }) catch {
        bridgeStringAlloc().free(col_name);
        return errors.jsError(ctx, .out_of_memory, "target(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_encode(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "encode() requires a column name", null, null, null);
    const col_name = qjs.valueToStringAlloc(ctx, argv[0], bridgeStringAlloc()) catch return errors.jsError(ctx, .invalid_arg, "encode() requires a string column name", null, null, null);

    var is_label = false;
    if (argc > 1) {
        const strategy = qjs.valueToStringAlloc(ctx, argv[1], bridgeStringAlloc()) catch null;
        defer if (strategy) |s| bridgeStringAlloc().free(s);
        if (strategy) |s| {
            if (std.mem.eql(u8, s, "label")) is_label = true;
        }
    }

    const step: pipeline.Step = if (is_label) .{ .encode_label = col_name } else .{ .encode_onehot = col_name };
    const new_ds = ds.addStep(step) catch {
        bridgeStringAlloc().free(col_name);
        return errors.jsError(ctx, .out_of_memory, "encode(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_clip(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 3) return errors.jsError(ctx, .invalid_arg, "clip() requires at least (column, min, max)", null, null, null);

    var min_val: f64 = 0;
    var max_val: f64 = 0;
    const min_idx: usize = @intCast(argc - 2);
    const max_idx: usize = @intCast(argc - 1);
    if (qjs.c.JS_ToFloat64(ctx, &min_val, argv[min_idx]) < 0 or qjs.c.JS_ToFloat64(ctx, &max_val, argv[max_idx]) < 0) {
        return errors.jsError(ctx, .invalid_arg, "clip() requires numeric min/max", null, null, null);
    }

    const num_cols: usize = @intCast(argc - 2);
    const cols = bridgeStringAlloc().alloc([]const u8, num_cols) catch return errors.jsError(ctx, .out_of_memory, "clip(): out of memory", error.OutOfMemory, null, null);
    for (0..num_cols) |i| {
        cols[i] = qjs.valueToStringAlloc(ctx, argv[i], bridgeStringAlloc()) catch {
            for (0..i) |j| bridgeStringAlloc().free(cols[j]);
            bridgeStringAlloc().free(cols);
            return errors.jsError(ctx, .invalid_arg, "clip() column arguments must be strings", null, null, null);
        };
    }

    const step: pipeline.Step = .{ .clip = .{ .columns = cols, .min_val = min_val, .max_val = max_val } };
    const new_ds = ds.addStep(step) catch {
        freeStringSlice(cols);
        return errors.jsError(ctx, .out_of_memory, "clip(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_shuffle(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, _: c_int, _: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    const new_ds = ds.addStep(.{ .shuffle = {} }) catch return errors.jsError(ctx, .out_of_memory, "shuffle(): out of memory", error.OutOfMemory, null, null);
    return createDatasetObject(ctx, new_ds);
}

fn js_sample(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "sample() requires a number", null, null, null);
    var n_value: f64 = 0;
    if (qjs.c.JS_ToFloat64(ctx, &n_value, argv[0]) < 0 or !std.math.isFinite(n_value) or n_value < 0 or @floor(n_value) != n_value) {
        return errors.jsError(ctx, .invalid_arg, "sample() requires a non-negative integer", null, null, null);
    }
    const n: usize = @intFromFloat(n_value);
    const new_ds = ds.addStep(.{ .sample = n }) catch return errors.jsError(ctx, .out_of_memory, "sample(): out of memory", error.OutOfMemory, null, null);
    return createDatasetObject(ctx, new_ds);
}

fn js_fillna(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 2) return errors.jsError(ctx, .invalid_arg, "fillna() requires (column, value)", null, null, null);
    const col_name = qjs.valueToStringAlloc(ctx, argv[0], bridgeStringAlloc()) catch return errors.jsError(ctx, .invalid_arg, "fillna() requires a string column name", null, null, null);
    var value: f64 = 0;
    if (qjs.c.JS_ToFloat64(ctx, &value, argv[1]) < 0) {
        bridgeStringAlloc().free(col_name);
        return errors.jsError(ctx, .invalid_arg, "fillna() requires a numeric value", null, null, null);
    }
    const new_ds = ds.addStep(.{ .fillna = .{ .column = col_name, .value = value } }) catch {
        bridgeStringAlloc().free(col_name);
        return errors.jsError(ctx, .out_of_memory, "fillna(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_dropna(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    const cols = if (argc > 0)
        extractVarargs(ctx, argc, argv) orelse return errors.jsError(ctx, .invalid_arg, "dropna() arguments must be strings", null, null, null)
    else
        bridgeStringAlloc().alloc([]const u8, 0) catch return errors.jsError(ctx, .out_of_memory, "dropna(): out of memory", error.OutOfMemory, null, null);
    const new_ds = ds.addStep(.{ .dropna = cols }) catch {
        freeStringSlice(cols);
        return errors.jsError(ctx, .out_of_memory, "dropna(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_rename(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 2) return errors.jsError(ctx, .invalid_arg, "rename() requires (oldName, newName)", null, null, null);
    const old_name = qjs.valueToStringAlloc(ctx, argv[0], bridgeStringAlloc()) catch return errors.jsError(ctx, .invalid_arg, "rename() requires string arguments", null, null, null);
    const new_name = qjs.valueToStringAlloc(ctx, argv[1], bridgeStringAlloc()) catch {
        bridgeStringAlloc().free(old_name);
        return errors.jsError(ctx, .invalid_arg, "rename() requires string arguments", null, null, null);
    };
    const new_ds = ds.addStep(.{ .rename = .{ .old_name = old_name, .new_name = new_name } }) catch {
        bridgeStringAlloc().free(old_name);
        bridgeStringAlloc().free(new_name);
        return errors.jsError(ctx, .out_of_memory, "rename(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_concat(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "concat() requires a Dataset argument", null, null, null);
    const other_ds = extractDataset(ctx, argv[0]) orelse return errors.jsError(ctx, .invalid_arg, "concat() requires a Dataset argument", null, null, null);

    const df_a = pipeline.execute(data_alloc, ds.shared.df, ds.steps.items) catch |err| return errors.jsNativeError(ctx, .internal, "concat(): pipeline failed", err, null);
    defer df_a.deinit();
    const df_b = pipeline.execute(data_alloc, other_ds.shared.df, other_ds.steps.items) catch |err| return errors.jsNativeError(ctx, .internal, "concat(): pipeline failed", err, null);
    defer df_b.deinit();
    const merged = pipeline.concatDataFrames(data_alloc, df_a, df_b) catch |err| return errors.jsError(ctx, errors.classifyNativeError(err), "concat(): column mismatch or out of memory", err, null, @errorReturnTrace());
    const result_ds = Dataset.init(merged) catch {
        merged.deinit();
        return errors.jsError(ctx, .out_of_memory, "concat(): out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, result_ds);
}

fn js_split(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "split() requires at least one ratio", null, null, null);
    if (argc > 8) return errors.jsError(ctx, .invalid_arg, "split() supports at most 8 ratios", null, null, null);

    var ratios: [8]f64 = undefined;
    var ratio_sum: f64 = 0;
    for (0..@as(usize, @intCast(argc))) |i| {
        if (qjs.c.JS_ToFloat64(ctx, &ratios[i], argv[i]) < 0 or ratios[i] <= 0 or ratios[i] >= 1) {
            return errors.jsError(ctx, .invalid_arg, "split() each ratio must be between 0 and 1", null, null, null);
        }
        ratio_sum += ratios[i];
    }
    if (ratio_sum > 1.0) return errors.jsError(ctx, .invalid_arg, "split() ratios must sum to at most 1", null, null, null);

    const df = pipeline.execute(data_alloc, ds.shared.df, ds.steps.items) catch |err| return errors.jsNativeError(ctx, .internal, "split(): pipeline failed", err, null);
    defer df.deinit();

    var boundaries: [8]usize = undefined;
    const total: f64 = @floatFromInt(df.num_rows);
    var cumulative: f64 = 0;
    for (0..@as(usize, @intCast(argc))) |i| {
        cumulative += ratios[i];
        boundaries[i] = @intFromFloat(@round(total * cumulative));
    }

    const parts = pipeline.splitDataFrame(data_alloc, df, boundaries[0..@intCast(argc)]) catch return errors.jsError(ctx, .out_of_memory, "split(): out of memory", error.OutOfMemory, null, null);
    defer data_alloc.free(parts);

    const result = qjs.c.JS_NewArray(ctx);
    if (qjs.isException(result)) return result;
    for (parts, 0..) |part_df, i| {
        const part_ds = Dataset.init(part_df) catch {
            for (parts[i..]) |p| p.deinit();
            return errors.jsError(ctx, .out_of_memory, "split(): out of memory", error.OutOfMemory, null, null);
        };
        if (qjs.c.JS_DefinePropertyValueUint32(ctx, result, @intCast(i), createDatasetObject(ctx, part_ds), qjs.c.JS_PROP_C_W_E) < 0) {
            return qjs.exceptionValue();
        }
    }
    return result;
}

const StepTag = enum { select, drop, normalize, standardize, log_transform, features };

fn pipelineStringArgs(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst, tag: StepTag) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    if (argc < 1) return errors.jsError(ctx, .invalid_arg, "expected string arguments", null, null, null);
    const cols = extractVarargs(ctx, argc, argv) orelse return errors.jsError(ctx, .invalid_arg, "expected string arguments", null, null, null);
    const step: pipeline.Step = switch (tag) {
        .select => .{ .select = cols },
        .drop => .{ .drop = cols },
        .normalize => .{ .normalize = cols },
        .standardize => .{ .standardize = cols },
        .log_transform => .{ .log_transform = cols },
        .features => .{ .features = cols },
    };
    const new_ds = ds.addStep(step) catch {
        freeStringSlice(cols);
        return errors.jsError(ctx, .out_of_memory, "dataset op: out of memory", error.OutOfMemory, null, null);
    };
    return createDatasetObject(ctx, new_ds);
}

fn js_toTensor(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    const dtype = parseToTensorDType(ctx, argc, argv) catch |err| switch (err) {
        error.InvalidArgument => return errors.jsError(ctx, .invalid_arg, "toTensor(): dtype must be 'f32' or 'f64'", null, null, null),
        error.OutOfMemory => return errors.jsError(ctx, .out_of_memory, "toTensor(): out of memory", err, null, null),
    };
    const df = pipeline.execute(data_alloc, ds.shared.df, ds.steps.items) catch |err| return errors.jsNativeError(ctx, .internal, "toTensor(): pipeline failed", err, null);
    defer df.deinit();
    if (df.num_rows == 0) return errors.jsError(ctx, .invalid_arg, "toTensor(): no data", null, null, null);

    const col_indices = data_alloc.alloc(usize, df.numCols()) catch return errors.jsError(ctx, .out_of_memory, "toTensor(): out of memory", error.OutOfMemory, null, null);
    defer data_alloc.free(col_indices);
    for (0..df.numCols()) |i| col_indices[i] = i;

    const flat = pipeline.toFlatBuffer(data_alloc, df, col_indices) catch return errors.jsError(ctx, .out_of_memory, "toTensor(): out of memory", error.OutOfMemory, null, null);
    var schema = pipeline.buildSchema(data_alloc, df, col_indices) catch {
        data_alloc.free(flat.data);
        return errors.jsError(ctx, .out_of_memory, "toTensor(): out of memory", error.OutOfMemory, null, null);
    };
    defer schema.deinit();

    const tensor_val = createTensorFromFlat(ctx, flat.data, df.num_rows, flat.width, dtype) catch return qjs.exceptionValue();
    const result = qjs.newObject(ctx);
    if (qjs.isException(result)) return result;
    qjs.setProperty(ctx, result, "data", tensor_val) catch return qjs.exceptionValue();
    qjs.setProperty(ctx, result, "schema", schemaToJS(ctx, &schema)) catch return qjs.exceptionValue();
    return result;
}

fn js_toTensors(ctx: ?*qjs.c.JSContext, this: qjs.c.JSValueConst, argc: c_int, argv: [*c]qjs.c.JSValueConst) callconv(.c) qjs.c.JSValue {
    const ds = extractDataset(ctx, this) orelse return errors.jsError(ctx, .invalid_arg, "invalid Dataset", null, null, null);
    const dtype = parseToTensorDType(ctx, argc, argv) catch |err| switch (err) {
        error.InvalidArgument => return errors.jsError(ctx, .invalid_arg, "toTensors(): dtype must be 'f32' or 'f64'", null, null, null),
        error.OutOfMemory => return errors.jsError(ctx, .out_of_memory, "toTensors(): out of memory", err, null, null),
    };

    var feature_cols: ?[]const []const u8 = null;
    var target_col: ?[]const u8 = null;
    for (ds.steps.items) |step| switch (step) {
        .features => |cols| feature_cols = cols,
        .target => |col| target_col = col,
        else => {},
    };
    if (feature_cols == null or target_col == null) return errors.jsError(ctx, .invalid_arg, "toTensors() requires .features() and .target() to be set", null, null, null);

    const df = pipeline.execute(data_alloc, ds.shared.df, ds.steps.items) catch |err| return errors.jsNativeError(ctx, .internal, "toTensors(): pipeline failed", err, null);
    defer df.deinit();
    if (df.num_rows == 0) return errors.jsError(ctx, .invalid_arg, "toTensors(): no data", null, null, null);

    const feat_indices = data_alloc.alloc(usize, feature_cols.?.len) catch return errors.jsError(ctx, .out_of_memory, "toTensors(): out of memory", error.OutOfMemory, null, null);
    defer data_alloc.free(feat_indices);
    for (feature_cols.?, 0..) |name, i| feat_indices[i] = df.findColumn(name) orelse return errors.jsError(ctx, .invalid_arg, "feature column not found", null, null, null);
    const target_idx = df.findColumn(target_col.?) orelse return errors.jsError(ctx, .invalid_arg, "target column not found", null, null, null);
    const target_indices = [_]usize{target_idx};

    const x_flat = pipeline.toFlatBuffer(data_alloc, df, feat_indices) catch return errors.jsError(ctx, .out_of_memory, "toTensors(): out of memory", error.OutOfMemory, null, null);
    const x_tensor = createTensorFromFlat(ctx, x_flat.data, df.num_rows, x_flat.width, dtype) catch return qjs.exceptionValue();
    const y_flat = pipeline.toFlatBuffer(data_alloc, df, &target_indices) catch return errors.jsError(ctx, .out_of_memory, "toTensors(): out of memory", error.OutOfMemory, null, null);
    const y_tensor = if (y_flat.width == 1)
        createTensorFromFlat1D(ctx, y_flat.data, df.num_rows, dtype) catch return qjs.exceptionValue()
    else
        createTensorFromFlat(ctx, y_flat.data, df.num_rows, y_flat.width, dtype) catch return qjs.exceptionValue();

    var schema = pipeline.buildSchema(data_alloc, df, feat_indices) catch return errors.jsError(ctx, .out_of_memory, "toTensors(): out of memory", error.OutOfMemory, null, null);
    defer schema.deinit();

    const result = qjs.newObject(ctx);
    if (qjs.isException(result)) return result;
    qjs.setProperty(ctx, result, "X", x_tensor) catch return qjs.exceptionValue();
    qjs.setProperty(ctx, result, "y", y_tensor) catch return qjs.exceptionValue();
    qjs.setProperty(ctx, result, "schema", schemaToJS(ctx, &schema)) catch return qjs.exceptionValue();
    return result;
}

fn createTensorFromFlat(ctx: ?*qjs.c.JSContext, data: []f64, num_rows: usize, width: usize, dtype: DType) !qjs.c.JSValue {
    if (false) return error.OutOfMemory;
    defer data_alloc.free(data);
    const device: compute.Device = switch (cfg.getDefaultDevice()) {
        .cpu => .cpu,
        .metal => .metal,
        .cuda => .cuda,
    };
    const value = compute.Tensor.createContiguous(data_alloc, &.{ num_rows, width }, dtype, device, false) catch return error.OutOfMemory;
    errdefer value.deinit();
    try writeFlatData(value, data, dtype);
    return tensor_binding.createTensorObject(ctx, value);
}

fn createTensorFromFlat1D(ctx: ?*qjs.c.JSContext, data: []f64, num_rows: usize, dtype: DType) !qjs.c.JSValue {
    if (false) return error.OutOfMemory;
    defer data_alloc.free(data);
    const device: compute.Device = switch (cfg.getDefaultDevice()) {
        .cpu => .cpu,
        .metal => .metal,
        .cuda => .cuda,
    };
    const value = compute.Tensor.createContiguous(data_alloc, &.{num_rows}, dtype, device, false) catch return error.OutOfMemory;
    errdefer value.deinit();
    try writeFlatData(value, data, dtype);
    return tensor_binding.createTensorObject(ctx, value);
}

fn writeFlatData(value: *compute.Tensor, data: []const f64, dtype: DType) !void {
    switch (dtype) {
        .f64 => try value.storage.?.writeFromHost(std.mem.sliceAsBytes(data)),
        .f32 => {
            const host = try data_alloc.alignedAlloc(f32, .@"8", data.len);
            defer data_alloc.free(host);
            for (data, 0..) |v, i| host[i] = @floatCast(v);
            try value.storage.?.writeFromHost(std.mem.sliceAsBytes(host));
        },
        .i64 => {
            const host = try data_alloc.alignedAlloc(i64, .@"8", data.len);
            defer data_alloc.free(host);
            for (data, 0..) |v, i| host[i] = @intFromFloat(v);
            try value.storage.?.writeFromHost(std.mem.sliceAsBytes(host));
        },
    }
}

fn schemaToJS(ctx: ?*qjs.c.JSContext, schema: *pipeline.SchemaResult) qjs.c.JSValue {
    const obj = qjs.newObject(ctx);
    if (qjs.isException(obj)) return obj;
    for (schema.entries) |entry| {
        const arr = qjs.c.JS_NewArray(ctx);
        if (qjs.isException(arr)) return qjs.exceptionValue();
        for (entry.indices, 0..) |idx, i| {
            if (qjs.c.JS_DefinePropertyValueUint32(ctx, arr, @intCast(i), qjs.c.JS_NewInt64(ctx, @intCast(idx)), qjs.c.JS_PROP_C_W_E) < 0) return qjs.exceptionValue();
        }
        const entry_name = bridgeStringAlloc().dupeZ(u8, entry.name) catch return qjs.exceptionValue();
        defer bridgeStringAlloc().free(entry_name);
        qjs.setProperty(ctx, obj, entry_name, arr) catch return qjs.exceptionValue();
    }
    return obj;
}

fn extractVarargs(ctx: ?*qjs.c.JSContext, argc: c_int, argv: [*c]qjs.c.JSValueConst) ?[][]const u8 {
    const count: usize = @intCast(argc);
    const cols = bridgeStringAlloc().alloc([]const u8, count) catch return null;
    for (0..count) |i| {
        cols[i] = qjs.valueToStringAlloc(ctx, argv[i], bridgeStringAlloc()) catch {
            for (0..i) |j| bridgeStringAlloc().free(cols[j]);
            bridgeStringAlloc().free(cols);
            return null;
        };
    }
    return cols;
}

fn makeStringArray(ctx: ?*qjs.c.JSContext, items: []const []const u8) ?qjs.c.JSValue {
    const arr = qjs.c.JS_NewArray(ctx);
    if (qjs.isException(arr)) return null;
    for (items, 0..) |item, i| {
        if (qjs.c.JS_DefinePropertyValueUint32(ctx, arr, @intCast(i), qjs.createString(ctx, item), qjs.c.JS_PROP_C_W_E) < 0) return null;
    }
    return arr;
}
