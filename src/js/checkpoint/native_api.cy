import compute_core.lib as compute;
import { DType } from compute_core.lib;
import std.fs;
import std.json;
import hao.js;
import .support.program as graph;
import .js.compute.native_api as compute_host;

struct CheckpointTensor {
    name: string;
    dtype: DType;
    dimensions: List<usize>;
    start: usize;
    end: usize;
}

fn argument(arguments_pointer: *const js.JsValue, index: usize): js.JsValue {
    unchecked {
        return *(arguments_pointer + index);
    }
}

fn property_int(ctx: *c_void, object: js.JsValue, name: c_str): ?i64 {
    const value = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, value);
    return js.read_int(ctx, value);
}

fn push_text(output: &List<u8>, value: string): void {
    for byte in value.bytes() {
        output.push(byte);
    }
}

fn push_decimal(output: &List<u8>, value: usize, digits: usize): void {
    if value == 0 {
        if digits > 0 {
            output.push(48);
        }
        return;
    }
    var divisor: usize = 1;
    var remaining_digits = digits;
    while remaining_digits > 1 {
        divisor = divisor * 10;
        remaining_digits = remaining_digits - 1;
    }
    while divisor > 0 {
        output.push(48 + ((value / divisor) % 10) as u8);
        divisor = divisor / 10;
    }
}

fn usize_digits(value: usize): usize {
    var digits: usize = 1;
    var remaining = value;
    while remaining >= 10 {
        remaining = remaining / 10;
        digits = digits + 1;
    }
    return digits;
}

fn push_usize(output: &List<u8>, value: usize): void {
    push_decimal(output, value, usize_digits(value));
}

fn push_json_string(output: &List<u8>, value: string): void {
    output.push(34);
    for byte in value.bytes() {
        if byte == 34 or byte == 92 {
            output.push(92);
            output.push(byte);
        } else if byte == 10 {
            push_text(output, "\\n");
        } else if byte == 13 {
            push_text(output, "\\r");
        } else if byte == 9 {
            push_text(output, "\\t");
        } else if byte >= 32 {
            output.push(byte);
        }
    }
    output.push(34);
}

fn checkpoint_dtype_name(dtype: DType): string {
    return switch dtype {
        .bf16 => "BF16";
        .f32 => "F32";
        .f64 => "F64";
        .u32 => "U32";
        .i64 => "I64";
    };
}

fn checkpoint_dtype(value: string): ?DType {
    if value == "BF16" {
        return DType.bf16;
    }
    if value == "F32" {
        return DType.f32;
    }
    if value == "F64" {
        return DType.f64;
    }
    if value == "U32" {
        return DType.u32;
    }
    if value == "I64" {
        return DType.i64;
    }
    return none;
}

fn tensor_spec(selected_dtype: DType,
    dimensions: []const usize
): compute.TensorSpec ! compute.ComputeError {
    return switch selected_dtype {
        .f32 => try compute.f32_spec(dimensions);
        .f64 => try compute.f64_spec(dimensions);
        .u32 => try compute.u32_spec(dimensions);
        .i64 => try compute.i64_spec(dimensions);
        .bf16 => try compute.bf16_spec(dimensions);
    };
}

fn save_native(ctx: *c_void, _: js.JsValue, argument_count: c_int,
    arguments_pointer: *const js.JsValue
): js.JsValue {
    if argument_count < 2 or
    js.is_array(argument(arguments_pointer, 0)) == 0 {
        return js.type_error(
            ctx,
            "checkpoint.save expects entries and a path"
        );
    }
    const path = js.read_string(ctx, argument(arguments_pointer, 1));
    if path is none {
        return js.type_error(
            ctx,
            "checkpoint.save expects a string path"
        );
    }
    const entries_value = argument(arguments_pointer, 0);
    const count = property_int(ctx, entries_value, "length");
    if count is none {
        return js.type_error(ctx, "checkpoint.save requires non-empty entries");
    }
    if count <= 0 {
        return js.type_error(ctx, "checkpoint.save requires non-empty entries");
    }
    var names: List<string> = list<string>();
    var tensors: List<usize> = list<usize>();
    var index: usize = 0;
    while index < count as usize {
        const item = js.get_property_uint32(
            ctx,
            entries_value,
            index as c_uint
        );
        defer js.free_value(ctx, item);
        const name_value = js.get_property_str(ctx, item, "name");
        defer js.free_value(ctx, name_value);
        const name = js.read_string(ctx, name_value);
        if name is none {
            return js.type_error(
                ctx,
                "checkpoint entry name must be a string"
            );
        }
        const tensor_value_js = js.get_property_str(ctx, item, "value");
        defer js.free_value(ctx, tensor_value_js);
        const tensor_handle = compute_host.live_tensor_handle(ctx, tensor_value_js);
        if tensor_handle is none {
            return js.type_error(
                ctx,
                "checkpoint entry value must be a Tensor"
            );
        }
        names.push(name);
        tensors.push(tensor_handle);
        index = index + 1;
    }
    var header: List<u8> = list<u8>();
    header.push(123);
    var offset: usize = 0;
    index = 0;
    while index < names.len() {
        if index != 0 {
            header.push(44);
        }
        const tensor = compute_host.tensor_for_handle(tensors[index]);
        push_json_string(&header, names[index]);
        push_text(&header, ":{\"dtype\":\"");
        push_text(&header, checkpoint_dtype_name(
            tensor.spec.dtype
        ));
        push_text(&header, "\",\"shape\":[");
        var dimension_index: usize = 0;
        while dimension_index < tensor.spec.dimensions.len() {
            if dimension_index != 0 {
                header.push(44);
            }
            push_usize(&header, tensor.spec.dimensions[dimension_index]);
            dimension_index = dimension_index + 1;
        }
        push_text(&header, "],\"data_offsets\":[");
        push_usize(&header, offset);
        const width = graph.dtype_width(tensor.spec.dtype);
        const count_value = tensor.spec.element_count() catch (error_code) {
            const ignored = error_code;
            return js.type_error(ctx, "checkpoint tensor shape is invalid");
        };
        offset = offset + count_value * width;
        header.push(44);
        push_usize(&header, offset);
        push_text(&header, "]}");
        index = index + 1;
    }
    header.push(125);
    var output: List<u8> = list<u8>();
    var prefix_index: usize = 0;
    while prefix_index < 8 {
        output.push(0);
        prefix_index = prefix_index + 1;
    }
    for byte in header[..] {
        output.push(byte);
    }
    index = 0;
    while index < tensors.len() {
        const tensor_data = graph.tensor_bytes(compute_host.tensor_for_handle(tensors[index]));
        for byte in tensor_data[..] {
            output.push(byte);
        }
        index = index + 1;
    }
    prefix_index = 0;
    var header_size = header.len();
    while prefix_index < 8 {
        output[prefix_index] = (header_size % 256) as u8;
        header_size = header_size / 256;
        prefix_index = prefix_index + 1;
    }
    fs.write_bytes(path, output[..]) catch (error_code) {
        const ignored = error_code;
        return js.type_error(
            ctx,
            "checkpoint.save failed to write file"
        );
    };
    return js.undefined();
}

fn little_u64(bytes: []const u8, start: usize): ?u64 {
    if start + 8 > bytes.len() {
        return none;
    }
    var value: u64 = 0;
    var index: usize = 0;
    while index < 8 {
        value = value | ((bytes[start + index] as u64) << (index * 8));
        index = index + 1;
    }
    return value;
}

fn parse_nonnegative(text: string, maximum: usize): ?usize {
    const bytes = text.bytes();
    if bytes.len() == 0 {
        return none;
    }
    var value: usize = 0;
    for byte in bytes {
        if byte < 48 or byte > 57 {
            return none;
        }
        const digit = (byte - 48) as usize;
        if value > (maximum - digit) / 10 {
            return none;
        }
        value = value * 10 + digit;
    }
    return value;
}

fn checkpoint_entries(header: &const json.Value,
    raw_size: usize
): ?List<CheckpointTensor> {
    if header is .object(document) {
        var result: List<CheckpointTensor> = list<CheckpointTensor>();
        for member in document.members[..] {
            if member.key == "__metadata__" {
                if member.value is .object(metadata) {
                    for value in metadata.members[..] {
                        if value.value is .string(text) {
                            const ignored = text;
                        } else {
                            return none;
                        }
                    }
                } else {
                    return none;
                }
                continue;
            }
            if member.value is .object(metadata) {
                const dtype_index = metadata.find("dtype");
                const shape_index = metadata.find("shape");
                const offsets_index = metadata.find("data_offsets");
                if dtype_index is none {
                    return none;
                }
                if shape_index is none {
                    return none;
                }
                if offsets_index is none {
                    return none;
                }
                const dtype_value = &metadata.members[dtype_index].value;
                if dtype_value is .string(dtype_text) {
                    const dtype = checkpoint_dtype(dtype_text);
                    if dtype is none {
                        return none;
                    }
                    const shape_value = &metadata.members[shape_index].value;
                    if shape_value is .array(shape) {
                        var dimensions: List<usize> = list<usize>();
                        var elements: usize = 1;
                        const maximum: usize = (0 as usize) -% 1;
                        for dimension in shape.items[..] {
                            if dimension is .number(number) {
                                const parsed = parse_nonnegative(
                                    number.spelling,
                                    maximum
                                );
                                if parsed is none {
                                    return none;
                                }
                                if parsed != 0 and
                                elements > maximum / parsed {
                                    return none;
                                }
                                elements = elements * parsed;
                                dimensions.push(parsed);
                            } else {
                                return none;
                            }
                        }
                        const offsets_value = &metadata.members[offsets_index].value;
                        if offsets_value is .array(offsets) {
                            if offsets.items.len() != 2 {
                                return none;
                            }
                            if offsets.items[0] is .number(first) {
                                if offsets.items[1] is .number(second) {
                                    const start = parse_nonnegative(
                                        first.spelling,
                                        maximum
                                    );
                                    const end = parse_nonnegative(
                                        second.spelling,
                                        maximum
                                    );
                                    if start is none {
                                        return none;
                                    }
                                    if end is none {
                                        return none;
                                    }
                                    if end < start or end > raw_size {
                                        return none;
                                    }
                                    const width = graph.dtype_width(dtype);
                                    if elements > maximum / width or
                                    elements * width != end - start {
                                        return none;
                                    }
                                    result.push(CheckpointTensor {
                                        name: member.key,
                                        dtype: dtype,
                                        dimensions: dimensions,
                                        start: start,
                                        end: end
                                    });
                                } else {
                                    return none;
                                }
                            } else {
                                return none;
                            }
                        } else {
                            return none;
                        }
                    } else {
                        return none;
                    }
                } else {
                    return none;
                }
            } else {
                return none;
            }
        }
        return result;
    }
    return none;
}

fn checkpoint_result(ctx: *c_void, argument_count: c_int,
    arguments_pointer: *const js.JsValue, inspect: bool
): js.JsValue {
    if argument_count < 1 {
        return js.type_error(ctx, "checkpoint.load expects a path");
    }
    const path = js.read_string(ctx, argument(arguments_pointer, 0));
    if path is none {
        return js.type_error(ctx, "checkpoint.load expects a string path");
    }
    const bytes = fs.read_bytes(path) catch (error_code) {
        const ignored = error_code;
        return js.type_error(
            ctx,
            "checkpoint.load failed to open file"
        );
    };
    if bytes.len() < 8 {
        return js.type_error(ctx, "checkpoint.load file is too small");
    }
    const header_size_u64 = little_u64(bytes[..], 0);
    if header_size_u64 is none {
        return js.type_error(ctx, "checkpoint.load truncated header");
    }
    if header_size_u64 > 16 * 1024 * 1024 {
        return js.type_error(ctx, "checkpoint.load has an invalid header");
    }
    if header_size_u64 > (bytes.len() - 8) as u64 {
        return js.type_error(ctx, "checkpoint.load has an invalid header");
    }
    const header_size = header_size_u64 as usize;
    const header_text = string.from(bytes[8..8 + header_size]);
    const parsed = json.parse(header_text) catch (error_code) {
        const ignored = error_code;
        return js.type_error(ctx, "checkpoint.load has invalid metadata");
    };
    const raw_size = bytes.len() - 8 - header_size;
    const entries = checkpoint_entries(&parsed, raw_size);
    if entries is none {
        return js.type_error(ctx, "checkpoint.load has invalid metadata");
    }
    var requested: List<string> = list<string>();
    const selective = not inspect and argument_count > 1 and
    not js.is_undefined(argument(arguments_pointer, 1));
    if selective {
        const names_value = argument(arguments_pointer, 1);
        if js.is_array(names_value) == 0 {
            return js.type_error(
                ctx,
                "checkpoint.load names must be an array"
            );
        }
        const name_count = property_int(ctx, names_value, "length");
        if name_count is none {
            return js.type_error(ctx, "Invalid checkpoint names");
        }
        if name_count < 0 {
            return js.type_error(ctx, "Invalid checkpoint names");
        }
        var name_index: usize = 0;
        while name_index < name_count as usize {
            const item = js.get_property_uint32(
                ctx,
                names_value,
                name_index as c_uint
            );
            defer js.free_value(ctx, item);
            const name = js.read_string(ctx, item);
            if name is none {
                return js.type_error(
                    ctx,
                    "checkpoint names must be strings"
                );
            }
            for previous in requested[..] {
                if previous == name {
                    return js.type_error(ctx, "Duplicate checkpoint name");
                }
            }
            requested.push(name);
            name_index = name_index + 1;
        }
        for name in requested[..] {
            var found = false;
            for entry in entries[..] {
                if entry.name == name {
                    found = true;
                }
            }
            if not found {
                return js.type_error_string(
                    ctx,
                    $"Missing requested checkpoint tensor: {name}"
                );
            }
        }
    }
    const result = js.new_object(ctx);
    if js.is_exception(result) {
        return result;
    }
    for entry in entries[..] {
        if inspect {
            const info = js.new_object(ctx);
            const shape = js.new_array(ctx);
            var dimension_index: usize = 0;
            while dimension_index < entry.dimensions.len() {
                const ignored = js.set_property_uint32(
                    ctx,
                    shape,
                    dimension_index as c_uint,
                    js.new_number(
                        ctx,
                        entry.dimensions[dimension_index] as f64
                    )
                );
                dimension_index = dimension_index + 1;
            }
            const ignored_shape = js.set_property_str(
                ctx,
                info,
                "shape",
                shape
            );
            js.set_string(
                ctx,
                info,
                "dtype",
                checkpoint_dtype_name(entry.dtype)
            );
            if js.set_dynamic_value(ctx, result, entry.name, info) < 0 {
                return js.type_error(
                    ctx,
                    "checkpoint.inspect failed to create result"
                );
            }
            continue;
        }
        var wanted = not selective;
        if selective {
            for name in requested[..] {
                if name == entry.name {
                    wanted = true;
                }
            }
        }
        if not wanted {
            continue;
        }
        const tensor_spec = tensor_spec(entry.dtype, entry.dimensions[..])
        catch (error_code) {
            const ignored = error_code;
            return js.type_error(
                ctx,
                "checkpoint.load invalid Tensor spec"
            );
        };
        const tensor = compute.tensor_bytes(
            tensor_spec,
            bytes[8 + header_size + entry.start..8 + header_size + entry.end]
        ) catch (error_code) {
            const ignored = error_code;
            return js.type_error(
                ctx,
                "checkpoint.load invalid Tensor"
            );
        };
        const object = compute_host.detached_tensor(ctx, tensor);
        if js.set_dynamic_value(ctx, result, entry.name, object) < 0 {
            return js.type_error(
                ctx,
                "checkpoint.load failed to create result"
            );
        }
    }
    return result;
}

fn save_native_callback(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_this = this_value;
    return save_native(ctx, this_value, argument_count, arguments_pointer);
}

fn load_native_callback(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_this = this_value;
    return checkpoint_result(ctx, argument_count, arguments_pointer, false);
}

fn inspect_checkpoint_callback(ctx: *c_void, this_value: js.JsValue,
    argument_count: c_int, arguments_pointer: *const js.JsValue
): js.JsValue {
    const ignored_this = this_value;
    return checkpoint_result(ctx, argument_count, arguments_pointer, true);
}

pub fn add_exports(ctx: *c_void, module_ptr: *c_void): c_int {
    if js.add_module_export(ctx, module_ptr, "saveNative") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "loadNative") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "inspectCheckpoint") != 0 {
        return -1;
    }
    if js.add_module_export(ctx, module_ptr, "default") != 0 {
        return -1;
    }
    return 0;
}

fn install_function(ctx: *c_void, object: js.JsValue, name: c_str,
    callback: js.Callback, argument_count: c_int, module_ptr: *c_void,
    export_name: c_str
): c_int {
    const value = js.new_function(
        ctx, callback, name, argument_count, 0, 0
    );
    if js.is_exception(value) {
        print("affon checkpoint native module: C function creation failed\n");
        return -1;
    }
    if js.set_property_str(ctx, object, name, value) < 0 {
        print("affon checkpoint native module: property installation failed\n");
        return -1;
    }
    const exported = js.get_property_str(ctx, object, name);
    defer js.free_value(ctx, exported);
    if js.set_module_export(
        ctx, module_ptr, export_name, js.dup_value(ctx, exported)
    ) < 0 {
        print("affon checkpoint native module: export installation failed\n");
        return -1;
    }
    return 0;
}

pub fn initialize(ctx: *c_void, module_ptr: *c_void): c_int {
    const object = js.new_object(ctx);
    if js.is_exception(object) {
        print("affon checkpoint native module: object creation failed\n");
        return -1;
    }
    if install_function(ctx, object, "saveNative", save_native_callback, 2,
        module_ptr, "saveNative"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "loadNative", load_native_callback, 2,
        module_ptr, "loadNative"
    ) != 0 {
        return -1;
    }
    if install_function(ctx, object, "inspectCheckpoint",
        inspect_checkpoint_callback, 1, module_ptr, "inspectCheckpoint"
    ) != 0 {
        return -1;
    }
    if js.set_module_export(ctx, module_ptr, "default", object) < 0 {
        print("affon checkpoint native module: default export failed\n");
        return -1;
    }
    return 0;
}
