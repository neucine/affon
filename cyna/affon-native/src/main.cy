import compute_core.lib as compute;
import hao.native_api as hao;
import { Runtime } from hao.runtime;
import { read_file_limited, script_read_error } from hao.fs;
import { exists } from std.fs;
import { write_stderr } from hao.stderr;
import { install_from as install_std } from hao.std;
import { collect, sort } from hao.std.test_files;
import { args, env_get } from std.process;
import { join } from std.path;
import .native_module;

const version = "0.4.2";

fn usage(): void {
    write_stderr(
        "Usage:\n  affon <file.ts|file.js>\n  affon run <file.ts|file.js>\n  affon test [--grep pattern] <file.ts|directory>...\n  affon --version\n"
    );
}

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

fn source_root(): ?string {
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

fn hao_modules_root(root: string): ?string {
    var candidate = join2(root.clone(), "../hao/build/modules");
    if exists(candidate.clone()) catch (code) => false {
        return candidate;
    }
    candidate = join2(root.clone(), "../hao/src/.cyna-cache/modules");
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

fn register_affon(runtime: &const Runtime, root: string): i32 {
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
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:_internal/compute/program",
        "src/js/compute/program.ts"
    ) != 0 { return 1; }
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
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:checkpoint",
        "src/js/checkpoint/index.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset",
        "src/js/dataset/index.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/tabular.ts",
        "src/js/dataset/tabular.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/text.ts",
        "src/js/dataset/text.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/text_io.ts",
        "src/js/dataset/text_io.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/text_loader.ts",
        "src/js/dataset/text_loader.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/text_utils.ts",
        "src/js/dataset/text_utils.ts"
    ) != 0 { return 1; }
    if register_file(runtime, root.clone(), "affon:dataset/token_windows.ts",
        "src/js/dataset/token_windows.ts"
    ) != 0 { return 1; }
    return register_file(runtime, root, "affon:dataset/tokenizer.ts",
        "src/js/dataset/tokenizer.ts"
    );
}

fn run_file(runtime: &const Runtime, path: string, test_file: bool): i32 {
    const source = read_file_limited(path.clone(), 32 * 1024 * 1024);
    if source is content {
        if test_file {
            return runtime.eval_test(content, path);
        }
        return runtime.eval(content, path);
    }
    return 1;
}

fn run_tests(runtime: &const Runtime, arguments: List<string>): i32 {
    var grep = "";
    var paths: List<string> = list<string>();
    var index: usize = 2;
    while index < arguments.len() {
        if arguments[index] == "--grep" {
            if index + 1 >= arguments.len() {
                usage();
                return 2;
            }
            grep = arguments[index + 1].clone();
            index = index + 2;
        } else {
            paths.push(arguments[index].clone());
            index = index + 1;
        }
    }
    if paths.len() == 0 {
        usage();
        return 2;
    }
    var files: List<string> = list<string>();
    for path in paths[..] {
        if collect(path.clone(), &files) != 0 {
            write_stderr("Unable to discover test files\n");
            return 1;
        }
    }
    sort(&files);
    if runtime.eval(
        "globalThis.__hao_test_started_ns = __hao_telemetry_monotonicNs();",
        "<affon-test-timing>"
    ) != 0 { return 1; }
    if runtime.eval("import 'std:test';", "<affon-test-setup>") != 0 {
        return 1;
    }
    var loaded: usize = 0;
    var failures: usize = 0;
    for path in files[..] {
        const status = run_file(runtime, path.clone(), true);
        if status != 0 or runtime.drain_test_jobs(path.clone()) < 0 {
            failures = failures + 1;
        } else {
            loaded = loaded + 1;
        }
    }
    if runtime.set_test_file_counts(loaded, failures, files.len()) != 0 {
        return 1;
    }
    if runtime.start_tests(grep) != 0 {
        return 1;
    }
    if runtime.run_until_idle() != 0 {
        return 1;
    }
    if failures != 0 or runtime.test_failures() != 0 {
        return 1;
    }
    return 0;
}

fn main(): i32 {
    if compute.api_version() != 1 or hao.api_version() != 1 {
        return 2;
    }
    const arguments = args();
    if arguments.len() < 2 {
        usage();
        return 2;
    }
    if arguments.len() == 2 and
        (arguments[1] == "--version" or arguments[1] == "version") {
        print($"affon {version}\n");
        return 0;
    }
    const root = source_root();
    if root is root_path {
        const runtime = Runtime.init();
        if runtime is host {
            defer host.deinit();
            const modules_root = hao_modules_root(root_path);
            if modules_root is modules_path {
                if install_std(&host, modules_path) != 0 {
                    write_stderr("Unable to install Hao standard modules\n");
                    return 1;
                }
                if native_module.register(&host) != 0 {
                    write_stderr("Unable to register the Affon Cyna native module\n");
                    return 1;
                }
                if register_affon(&host, root_path) != 0 {
                    return 1;
                }
                if arguments[1] == "test" {
                    return run_tests(&host, arguments);
                }
                if arguments[1] == "run" {
                    if arguments.len() != 3 {
                        usage();
                        return 2;
                    }
                    if run_file(&host, arguments[2].clone(), false) != 0 {
                        return 1;
                    }
                } else {
                    if arguments.len() != 2 {
                        usage();
                        return 2;
                    }
                    if run_file(&host, arguments[1].clone(), false) != 0 {
                        return 1;
                    }
                }
                return host.run_until_idle();
            }
            write_stderr("Unable to find Hao standard modules\n");
            return 1;
        }
        write_stderr("Unable to initialize the Hao runtime\n");
        return 1;
    }
    write_stderr(
        "Unable to find Affon source modules; set AFFON_ROOT or run from the Affon checkout\n"
    );
    return 1;
}
