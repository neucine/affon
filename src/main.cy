import { Runtime } from hao.runtime;
import { read_file_limited, script_read_error } from hao.fs;
import { write_stderr } from hao.stderr;
import { install_from as install_std } from hao.std;
import { collect, sort } from hao.std.test_files;
import { args } from std.process;
import .affon;

fn usage(): void {
    write_stderr(
        "Usage:\n  affon <file.ts|file.js>\n  affon run <file.ts|file.js>\n  affon test [--grep pattern] <file.ts|directory>...\n  affon --version\n"
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
    const reason = script_read_error(path, 32 * 1024 * 1024);
    if reason == 2 {
        write_stderr("NameTooLong\n");
    } else if reason == 1 {
        write_stderr("StreamTooLong\n");
    } else if reason == 0 {
        write_stderr("FileNotFound\n");
    } else {
        write_stderr("FileReadFailed\n");
    }
    if test_file {
        return 2;
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
    ) != 0 {
        return 1;
    }
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
    const arguments = args();
    if arguments.len() < 2 {
        usage();
        return 2;
    }
    if arguments.len() == 2 and
    (arguments[1] == "--version" or arguments[1] == "version") {
        print($"affon {affon.version}\n");
        return 0;
    }
    const root = affon.source_root();
    if root is root_path {
        const runtime = Runtime.init();
        if runtime is host {
            defer host.deinit();
            const modules_root = affon.hao_modules_root(root_path);
            if modules_root is modules_path {
                if install_std(&host, modules_path) != 0 {
                    write_stderr("Unable to install Hao standard modules\n");
                    return 1;
                }
                if affon.register(&host, root_path) != 0 {
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
