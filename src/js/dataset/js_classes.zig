pub const c = @import("hao").qjs.c;

var tensor_class_id: c.JSClassID = 0;
var tensor_class_id_ready = false;
var array_class_id: c.JSClassID = 0;
var array_class_id_ready = false;
var dataset_class_id: c.JSClassID = 0;
var dataset_class_id_ready = false;
var embedder_class_id: c.JSClassID = 0;
var embedder_class_id_ready = false;
var file_class_id: c.JSClassID = 0;
var file_class_id_ready = false;
var trace_handle_class_id: c.JSClassID = 0;
var trace_handle_class_id_ready = false;
var compiled_executable_class_id: c.JSClassID = 0;
var compiled_executable_class_id_ready = false;

pub fn ensureRegistered(rt: ?*anyopaque) void {
    _ = ensureTensorClassId(rt);
    _ = ensureArrayClassId(rt);
    _ = ensureDatasetClassId(rt);
    _ = ensureEmbedderClassId(rt);
    _ = ensureFileClassId(rt);
    _ = ensureTraceHandleClassId(rt);
    _ = ensureCompiledExecutableClassId(rt);
}

pub fn ensureTensorClassId(rt: ?*anyopaque) u32 {
    if (!tensor_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &tensor_class_id);
        tensor_class_id_ready = true;
    }
    return tensor_class_id;
}

pub fn ensureArrayClassId(rt: ?*anyopaque) u32 {
    if (!array_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &array_class_id);
        array_class_id_ready = true;
    }
    return array_class_id;
}

pub fn ensureDatasetClassId(rt: ?*anyopaque) u32 {
    if (!dataset_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &dataset_class_id);
        dataset_class_id_ready = true;
    }
    return dataset_class_id;
}

pub fn ensureEmbedderClassId(rt: ?*anyopaque) u32 {
    if (!embedder_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &embedder_class_id);
        embedder_class_id_ready = true;
    }
    return embedder_class_id;
}

pub fn ensureFileClassId(rt: ?*anyopaque) u32 {
    if (!file_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &file_class_id);
        file_class_id_ready = true;
    }
    return file_class_id;
}

pub fn ensureTraceHandleClassId(rt: ?*anyopaque) u32 {
    if (!trace_handle_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &trace_handle_class_id);
        trace_handle_class_id_ready = true;
    }
    return trace_handle_class_id;
}

pub fn ensureCompiledExecutableClassId(rt: ?*anyopaque) u32 {
    if (!compiled_executable_class_id_ready) {
        _ = c.JS_NewClassID(@ptrCast(rt), &compiled_executable_class_id);
        compiled_executable_class_id_ready = true;
    }
    return compiled_executable_class_id;
}
