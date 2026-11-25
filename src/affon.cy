import { Runtime } from hao.runtime;
import { read_file_limited, script_read_error } from hao.fs;
import { exists } from std.fs;
import { env_get } from std.process;
import { write_stderr } from hao.stderr;
import { join } from std.path;
import .js.checkpoint.native as checkpoint_native;
import .js.compute.native;

pub const version = "0.4.2";

fn join2(root: string, child: string): string {
    const parts: [2]string = [root.clone(), child];
    return join(parts[..]);
}

fn source_root_at(candidate: string): ?string {
    const probe = join2(candidate.clone(), "src/js/compute/public.ts");
    if exists(probe) catch (code) => false {
        return candidate;
    }
    return none;
}

pub fn source_root(): ?string {
    if env_get("AFFON_ROOT") is configured {
        if source_root_at(configured.clone()) is root {
            return root;
        }
    }
    if env_get("PWD") is current {
        if source_root_at(current.clone()) is root {
            return root;
        }
        var cursor = current.clone();
        var depth: usize = 0;
        while depth < 8 {
            cursor = join2(cursor, "..");
            if source_root_at(cursor.clone()) is root {
                return root;
            }
            depth = depth + 1;
        }
    }
    return none;
}

pub fn hao_modules_root(root: string): ?string {
    const candidate = join2(root.clone(), "../hao/build/modules");
    if exists(candidate.clone()) catch (code) => false {
        return candidate;
    }
    return none;
}

fn register_file(runtime: &const Runtime, root: string, specifier: string,
    relative_path: string
): i32 {
    const path = join2(root.clone(), relative_path);
    const source = read_file_limited(path.clone(), 32 * 1024 * 1024);
    if source is content {
        return runtime.register_source_module(specifier, content);
    }
    const reason = script_read_error(path, 32 * 1024 * 1024);
    if reason == 2 {
        write_stderr("error: source path is too long\n");
    } else if reason == 1 {
        write_stderr("error: source file is too large\n");
    } else if reason == 0 {
        write_stderr("error: source file is missing\n");
    } else {
        write_stderr("error: source file could not be read\n");
    }
    return 1;
}

fn register_modules(runtime: &const Runtime, root: string): i32 {
    const errors_path = join2(root.clone(), "src/js/errors-global.js");
    const errors_source = read_file_limited(errors_path, 32 * 1024 * 1024);
    if errors_source is source {
        if runtime.register_source_module(
            "affon:errors",
            $"{source}\nexport const AffonError = globalThis.AffonError;\nexport default AffonError;\n"
        ) != 0 {
            return 1;
        }
    } else {
        return 1;
    }
    if runtime.register_source_module(
        "affon:runtime",
        $"import 'affon:errors';\nexport const name = \"affon\";\nexport const version = \"{version}\";\n"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:core",
        "src/js/compute/public.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:_internal/compute/program",
        "src/js/compute/program.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:_internal/compute/program_wire",
        "src/js/compute/program_wire.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:ops", "src/js/ops.ts") != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:nn", "src/js/nn.ts") != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:optim", "src/js/optim.ts") != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:core/persistence.ts",
        "src/js/compute/persistence.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:checkpoint",
        "src/js/checkpoint/index.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset",
        "src/js/dataset/index.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/tabular.ts",
        "src/js/dataset/tabular.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/text.ts",
        "src/js/dataset/text.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/text_io.ts",
        "src/js/dataset/text_io.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/text_loader.ts",
        "src/js/dataset/text_loader.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/text_utils.ts",
        "src/js/dataset/text_utils.ts"
    ) != 0 {
        return 1;
    }
    if register_file(runtime, root.clone(), "affon:dataset/token_windows.ts",
        "src/js/dataset/token_windows.ts"
    ) != 0 {
        return 1;
    }
    return register_file(runtime, root, "affon:dataset/tokenizer.ts",
        "src/js/dataset/tokenizer.ts"
    );
}

pub fn register(runtime: &const Runtime, root: string): i32 {
    if native.register(runtime) != 0 {
        write_stderr("Unable to register the Affon compute native module\n");
        return 1;
    }
    if checkpoint_native.register(runtime) != 0 {
        write_stderr("Unable to register the Affon checkpoint native module\n");
        return 1;
    }
    return register_modules(runtime, root);
}
