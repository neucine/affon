// Narrow QuickJS boundary used by the Affon native module. The representation
// and exports are owned by the JavaScript engine; Affon's application-facing
// surface remains ordinary Cyna and Compute Core types.
pub c struct JSValue {
    pub payload: c_llong;
    pub tag: c_llong;
}

pub const tag_int: c_llong = 0;
pub const tag_bool: c_llong = 1;
pub const tag_null: c_llong = 2;
pub const tag_undefined: c_llong = 3;
pub const tag_exception: c_llong = 6;

pub fn undefined(): JSValue {
    return JSValue { payload: 0, tag: tag_undefined };
}

pub fn is_exception(value: JSValue): bool {
    return value.tag == tag_exception;
}

pub type Callback = c fn(*c_void, JSValue, c_int, *const JSValue): JSValue;

pub extern c("JS_NewCModule") fn new_c_module(context: *c_void,
    name: *const c_char, initialize: c fn(*c_void, *c_void): c_int
): *c_void;
pub extern c("JS_AddModuleExport") fn add_module_export(context: *c_void,
    module_ptr: *c_void, name: c_str
): c_int;
pub extern c("JS_SetModuleExport") fn set_module_export(context: *c_void,
    module_ptr: *c_void, name: c_str, value: JSValue
): c_int;
pub extern c("JS_NewCFunction2") fn new_c_function(context: *c_void,
    callback: Callback, name: c_str, length: c_int, kind: c_int, magic: c_int
): JSValue;
pub extern c("JS_NewObject") fn new_object(context: *c_void): JSValue;
pub extern c("JS_NewArray") fn new_array(context: *c_void): JSValue;
pub extern c("JS_NewNumber") fn new_number(context: *c_void, value: c_double): JSValue;
pub extern c("affon_js_new_number_bits") fn new_number_bits(
    context: *c_void,
    bits: c_llong
): JSValue;
pub fn new_number_from_f64(context: *c_void, value: f64): JSValue {
    const bits = value as! c_llong;
    return new_number_bits(context, bits);
}
pub extern c("JS_NewStringLen") fn new_string_len(context: *c_void,
    value: *const c_char, length: c_size_t
): JSValue;
pub extern c("JS_NewArrayBufferCopy") fn new_array_buffer_copy(context: *c_void,
    value: *const c_uchar, length: c_size_t
): JSValue;
pub extern c("JS_NewTypeError") fn new_type_error(context: *c_void,
    message: c_str
): JSValue;
pub extern c("JS_NewError") fn new_error(context: *c_void): JSValue;
pub extern c("JS_Throw") fn throw_value(context: *c_void, value: JSValue): JSValue;
pub extern c("JS_GetException") fn get_exception(context: *c_void): JSValue;
pub extern c("JS_SetPropertyStr") fn set_property_str(context: *c_void,
    object: JSValue, name: c_str, value: JSValue
): c_int;
pub extern c("JS_SetPropertyUint32") fn set_property_uint32(context: *c_void,
    object: JSValue, index: c_uint, value: JSValue
): c_int;
pub extern c("JS_GetPropertyStr") fn get_property_str(context: *c_void,
    object: JSValue, name: c_str
): JSValue;
pub extern c("JS_GetPropertyUint32") fn get_property_uint32(context: *c_void,
    object: JSValue, index: c_uint
): JSValue;
pub extern c("JS_NewAtomLen") fn new_atom_len(context: *c_void,
    value: *const c_char, length: c_size_t
): c_uint;
pub extern c("JS_SetProperty") fn set_property(context: *c_void,
    object: JSValue, property: c_uint, value: JSValue
): c_int;
pub extern c("JS_FreeAtom") fn free_atom(context: *c_void, atom: c_uint): void;
pub extern c("JS_GetArrayBuffer") fn get_array_buffer(context: *c_void,
    length: *c_size_t, value: JSValue
): ?*c_uchar;
pub extern c("JS_IsArray") fn is_array(value: JSValue): c_int;
pub extern c("JS_ToInt64") fn to_int64(context: *c_void, output: *c_llong,
    value: JSValue
): c_int;
pub extern c("JS_ToFloat64") fn to_float64(context: *c_void, output: *c_double,
    value: JSValue
): c_int;
pub extern c("JS_ToBool") fn to_bool(context: *c_void, value: JSValue): c_int;
pub extern c("JS_ToCStringLen2") fn to_c_string_len(context: *c_void,
    length: *c_size_t, value: JSValue, cesu8: c_int
): *const c_char;
pub extern c("JS_FreeCString") fn free_c_string(context: *c_void,
    value: *const c_char
): void;
pub extern c("JS_FreeValue") fn free_value(context: *c_void, value: JSValue): void;
pub extern c("JS_DupValue") fn dup_value(context: *c_void, value: JSValue): JSValue;

pub fn type_error(context: *c_void, message: c_str): JSValue {
    return throw_value(context, new_type_error(context, message));
}

pub fn type_error_string(context: *c_void, message: string): JSValue {
    const error = new_error(context);
    const ignored = set_property_str(
        context,
        error,
        "message",
        string(context, message)
    );
    return throw_value(context, error);
}

pub fn string(context: *c_void, value: string): JSValue {
    const bytes = value.bytes();
    return new_string_len(
        context,
        bytes.ptr as! *const c_char,
        bytes.len() as c_size_t
    );
}

pub fn set_number(context: *c_void, object: JSValue, name: c_str,
    value: f64
): void {
    const ignored = set_property_str(
        context, object, name, new_number_from_f64(context, value)
    );
}

pub fn set_string(context: *c_void, object: JSValue, name: c_str,
    value: string
): void {
    const ignored = set_property_str(
        context, object, name, string(context, value)
    );
}

pub fn set_dynamic_string(context: *c_void, object: JSValue, name: string,
    value: string
): c_int {
    const bytes = name.bytes();
    const atom = new_atom_len(
        context,
        bytes.ptr as! *const c_char,
        bytes.len() as c_size_t
    );
    defer free_atom(context, atom);
    return set_property(context, object, atom, string(context, value));
}

pub fn set_dynamic_value(context: *c_void, object: JSValue, name: string,
    value: JSValue
): c_int {
    const bytes = name.bytes();
    const atom = new_atom_len(
        context,
        bytes.ptr as! *const c_char,
        bytes.len() as c_size_t
    );
    defer free_atom(context, atom);
    return set_property(context, object, atom, value);
}

pub fn set_function(context: *c_void, object: JSValue, name: c_str,
    callback: Callback, argument_count: c_int
): void {
    const ignored = set_property_str(
        context, object, name,
        new_c_function(context, callback, name, argument_count, 0, 0)
    );
}

pub fn read_string(context: *c_void, value: JSValue): ?string {
    var length_storage: [1]c_size_t = [0];
    const pointer = to_c_string_len(context, length_storage[..].ptr, value, 0);
    const length = length_storage[0];
    if length > 16 * 1024 * 1024 {
        free_c_string(context, pointer);
        return none;
    }
    var bytes: List<u8> = list<u8>();
    var index: usize = 0;
    while index < length as usize {
        unchecked {
            bytes.push(*(pointer + index) as u8);
        }
        index = index + 1;
    }
    free_c_string(context, pointer);
    return string.from(bytes[..]);
}

pub fn read_int(context: *c_void, value: JSValue): ?i64 {
    var output_storage: [1]c_llong = [0];
    if to_int64(context, output_storage[..].ptr, value) != 0 {
        return none;
    }
    return output_storage[0] as i64;
}

pub fn read_number(context: *c_void, value: JSValue): ?f64 {
    var output_storage: [1]c_double = [0.0];
    if to_float64(context, output_storage[..].ptr, value) != 0 {
        return none;
    }
    return output_storage[0] as f64;
}
