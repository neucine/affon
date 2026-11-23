import hao.native_api as hao;
import hao.js;
import { Runtime } from hao.runtime;
import .js.compute.native_api as api;

fn load_module(context: *c_void, name: *const c_char): *c_void {
    const initializer = c fn (context_ptr: *c_void, module_ptr: *c_void): c_int {
        return api.initialize(context_ptr, module_ptr);
    };
    const module_ptr = js.new_module(context, name, initializer);
    if api.add_exports(context, module_ptr) != 0 {
        print("affon native module: export registration failed\n");
        return module_ptr;
    }
    return module_ptr;
}

pub fn register(runtime: &const Runtime): i32 {
    api.configure(runtime);
    return hao.register_native_module(runtime, "affon:core/native",
        c fn (context: *c_void, name: *const c_char): *c_void {
            return load_module(context, name);
        }
    );
}
