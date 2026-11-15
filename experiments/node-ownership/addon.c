#include <node_api.h>
#include <stdint.h>
#include <stddef.h>

typedef struct Handle Handle;
typedef struct Value Value;
extern Handle *affon_session_new(void);
extern void affon_session_close(Handle *);
extern void affon_session_destroy(Handle *);
extern Value *affon_tensor_new(Handle *, size_t);
extern void affon_tensor_close(Value *);
extern void affon_tensor_destroy(Value *);
extern size_t affon_active_bytes(Handle *);
extern size_t affon_live_tensors(Handle *);

static napi_value fail(napi_env env, const char *message) {
    napi_throw_error(env, NULL, message);
    return NULL;
}
static void session_finalizer(napi_env env, void *data, void *hint) {
    (void)env; (void)hint; affon_session_destroy((Handle *)data);
}
static void tensor_finalizer(napi_env env, void *data, void *hint) {
    (void)env; (void)hint; affon_tensor_destroy((Value *)data);
}
static void *argument(napi_env env, napi_callback_info info, size_t expected, napi_value *argv) {
    size_t count = expected;
    if (napi_get_cb_info(env, info, &count, argv, NULL, NULL) != napi_ok || count != expected)
        return NULL;
    void *pointer = NULL;
    if (napi_get_value_external(env, argv[0], &pointer) != napi_ok) return NULL;
    return pointer;
}
static napi_value open_session(napi_env env, napi_callback_info info) {
    (void)info;
    Handle *handle = affon_session_new();
    if (!handle) return fail(env, "could not create CPU session");
    napi_value result;
    if (napi_create_external(env, handle, session_finalizer, NULL, &result) != napi_ok) {
        affon_session_destroy(handle); return fail(env, "could not wrap session");
    }
    return result;
}
static napi_value close_session(napi_env env, napi_callback_info info) {
    napi_value argv[1]; Handle *handle = argument(env, info, 1, argv);
    if (!handle) return fail(env, "invalid session");
    affon_session_close(handle);
    napi_value result; napi_get_undefined(env, &result); return result;
}
static napi_value new_tensor(napi_env env, napi_callback_info info) {
    napi_value argv[2]; Handle *handle = argument(env, info, 2, argv);
    if (!handle) return fail(env, "invalid session");
    uint32_t count;
    if (napi_get_value_uint32(env, argv[1], &count) != napi_ok) return fail(env, "invalid count");
    Value *value = affon_tensor_new(handle, count);
    if (!value) return fail(env, "could not create tensor");
    napi_value result;
    if (napi_create_external(env, value, tensor_finalizer, NULL, &result) != napi_ok) {
        affon_tensor_destroy(value); return fail(env, "could not wrap tensor");
    }
    return result;
}
static napi_value close_tensor(napi_env env, napi_callback_info info) {
    napi_value argv[1]; Value *value = argument(env, info, 1, argv);
    if (!value) return fail(env, "invalid tensor");
    affon_tensor_close(value);
    napi_value result; napi_get_undefined(env, &result); return result;
}
static napi_value stats(napi_env env, napi_callback_info info) {
    napi_value argv[1]; Handle *handle = argument(env, info, 1, argv);
    if (!handle) return fail(env, "invalid session");
    napi_value result, bytes, tensors;
    napi_create_object(env, &result);
    napi_create_double(env, (double)affon_active_bytes(handle), &bytes);
    napi_create_double(env, (double)affon_live_tensors(handle), &tensors);
    napi_set_named_property(env, result, "activeBytes", bytes);
    napi_set_named_property(env, result, "liveTensors", tensors);
    return result;
}
NAPI_MODULE_INIT() {
    napi_property_descriptor methods[] = {
        {"openSession", NULL, open_session, NULL, NULL, NULL, napi_default, NULL},
        {"closeSession", NULL, close_session, NULL, NULL, NULL, napi_default, NULL},
        {"newTensor", NULL, new_tensor, NULL, NULL, NULL, napi_default, NULL},
        {"closeTensor", NULL, close_tensor, NULL, NULL, NULL, napi_default, NULL},
        {"stats", NULL, stats, NULL, NULL, NULL, napi_default, NULL},
    };
    napi_define_properties(env, exports, sizeof(methods) / sizeof(methods[0]), methods);
    return exports;
}
