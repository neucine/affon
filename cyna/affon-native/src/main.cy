import std.json;
import compute_core.lib as compute;
import hao.native_api as hao;
import .graph;
import .json_value as fields;
import .wire;

fn error_response(message: string): void ! wire.WireError {
    const empty: [0]u8 = [];
    var writer = wire.json_writer();
    writer.raw("{\"ok\":false,\"error\":");
    writer.string(message.bytes());
    writer.raw("}");
    try wire.write_frame(writer.finish(), empty[..]);
}

fn error_code_response(message: string, code: string): void ! wire.WireError {
    const empty: [0]u8 = [];
    var writer = wire.json_writer();
    writer.raw("{\"ok\":false,\"error\":");
    writer.string(message.bytes());
    writer.raw(",\"code\":");
    writer.string(code.bytes());
    writer.raw("}");
    try wire.write_frame(writer.finish(), empty[..]);
}

fn success_response(payload: []const u8): void ! wire.WireError {
    try wire.write_frame("{\"ok\":true}", payload);
}

fn dtype_name(value: compute.DType): string {
    return switch value {
        .f32 => "f32";
        .f64 => "f64";
        .i64 => "i64";
        .u32 => "u32";
        .bf16 => "bf16";
    };
}

fn input_bytes(compiled: &const graph.CompiledGraph,
    payload: []const u8
): ?List<compute.Tensor> {
    var inputs: List<compute.Tensor> = list<compute.Tensor>();
    var offset: usize = 0;
    for guard in compiled.executable.input_guards[..] {
        const spec = guard.spec;
        const count = spec.element_count() catch (code) {
            return none;
        };
        const bytes = count * graph.dtype_width(spec.dtype);
        if bytes > payload.len() - offset {
            return none;
        }
        var owned_spec = compute.f32_spec(spec.dimensions[..])
        catch (code) {
            return none;
        };
        owned_spec.dtype = spec.dtype;
        const value = compute.tensor_bytes(owned_spec, payload[offset..offset + bytes])
        catch (code) {
            return none;
        };
        inputs.push(value);
        offset = offset + bytes;
    }
    if offset != payload.len() {
        return none;
    }
    return inputs;
}

fn run_graph(compiled: &const graph.CompiledGraph,
    payload: []const u8
): bool ! wire.WireError {
    if input_bytes(compiled, payload) is inputs {
        const outputs = compute.run(&compiled.executable, inputs[..])
        catch (code) {
            try error_code_response("compute run failed", "InvalidCommand");
            return false;
        };
        var bytes: List<u8> = list<u8>();
        for output in outputs[..] {
            const encoded = graph.tensor_bytes(&output);
            for byte in encoded[..] {
                bytes.push(byte);
            }
        }
        try success_response(bytes.view());
        return true;
    }
    try error_code_response("input byte count or tensor spec mismatch",
        "InvalidShape"
    );
    return false;
}

struct Driver {
    current: ?graph.CompiledGraph;
    observed_run_count: usize;
    failed_run_count: usize;
}

fn evidence(state: &const Driver): void ! wire.WireError {
    const empty: [0]u8 = [];
    var writer = wire.json_writer();
    writer.raw("{\"ok\":true,\"backend\":\"cyna_cpu\",\"run_count\":");
    writer.usize(state.observed_run_count);
    writer.raw(",\"failed_run_count\":");
    writer.usize(state.failed_run_count);
    writer.raw(",\"output_count\":");
    var output_count: usize = 0;
    if state.current is compiled {
        output_count = compiled.outputs.len();
    }
    writer.usize(output_count);
    writer.raw("}");
    try wire.write_frame(writer.finish(), empty[..]);
}

fn dispatch(document: &const json.Value, payload: []const u8,
    state: &Driver
): bool ! wire.WireError {
    const empty: [0]u8 = [];
    if document is .object(request) {
        if fields.string_field(&request, "command") is command {
            if command == "shutdown" {
                try success_response(empty[..]);
                return false;
            }
            if command == "compile" {
                if fields.field_index(&request, "program") is index {
                    var goal = "balanced";
                    if fields.string_field(&request, "goal") is requested_goal {
                        goal = requested_goal;
                    }
                    const next = graph.compile_program(&request.members[index].value, goal)
                    catch (code) {
                        try error_response("compute compilation failed");
                        return true;
                    };
                    const output_count = next.outputs.len();
                    state.current = next;
                    var writer = wire.json_writer();
                    writer.raw("{\"ok\":true,\"input_count\":");
                    if state.current is compiled {
                        writer.usize(compiled.executable.input_guards.len());
                    } else {
                        writer.usize(0);
                    }
                    writer.raw(",\"outputs\":[");
                    var output_index: usize = 0;
                    if state.current is compiled {
                        while output_index < output_count {
                            if output_index != 0 {
                                writer.raw(",");
                            }
                            const output = &compiled.outputs[output_index];
                            writer.raw("{\"dtype\":\"");
                            writer.raw(dtype_name(output.dtype));
                            writer.raw("\",\"shape\":[");
                            var dimension_index: usize = 0;
                            while dimension_index < output.dimensions.len() {
                                if dimension_index != 0 {
                                    writer.raw(",");
                                }
                                writer.usize(output.dimensions[dimension_index]);
                                dimension_index = dimension_index + 1;
                            }
                            writer.raw("]}");
                            output_index = output_index + 1;
                        }
                    }
                    writer.raw("],\"output_count\":");
                    writer.usize(output_count);
                    writer.raw("}");
                    try wire.write_frame(writer.finish(), empty[..]);
                    return true;
                }
                try error_response("missing program");
                return true;
            }
            if state.current is compiled {
                if command == "run" {
                    const succeeded = try run_graph(&compiled, payload);
                    if succeeded {
                        state.observed_run_count = state.observed_run_count + 1;
                    } else {
                        state.failed_run_count = state.failed_run_count + 1;
                    }
                    return true;
                }
                if command == "evidence" {
                    try evidence(state);
                    return true;
                }
            }
            try error_response("unknown command or no compiled graph");
            return true;
        }
    }
    try error_response("invalid request");
    return true;
}

fn main(): i32 {
    if compute.api_version() != 1 or hao.api_version() != 1 {
        return 2;
    }
    var state = Driver {
        current: none,
        observed_run_count: 0,
        failed_run_count: 0
    };
    while true {
        const frame = wire.read_frame() catch (code) {
            return 3;
        };
        if frame is none {
            break;
        }
        const bytes = frame as List<u8>;
        if wire.metadata_length(bytes[..]) is metadata_length {
            const metadata = json.parse_bytes(bytes[4..4 + metadata_length])
            catch (code) {
                error_response("invalid JSON metadata") catch (write_code) {
                    return 4;
                };
                continue;
            };
            const keep_running = dispatch(&metadata,
                bytes[4 + metadata_length..], &state
            )
            catch (code) {
                return 5;
            };
            if not keep_running {
                break;
            }
        } else {
            error_response("invalid frame") catch (code) {
                return 6;
            };
        }
    }
    return 0;
}
