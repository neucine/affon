const std = @import("std");
const hao = @import("hao");
pub const config = @import("config.zig");
pub const memory = @import("memory.zig");
const compute_native = @import("js/compute/native_v2.zig");
const dataset_native = @import("js/dataset/native.zig");
const qjs = hao.qjs;
const packages = hao.package;

pub const package_name = "affon";
pub const version = "0.3.0";
pub const compute = @import("compute");

comptime { _ = config; _ = memory; }

const sources = [_]hao.SourceModule{
    .{ .specifier = "affon:errors", .source = @embedFile("js/errors-global.js") ++ "\nexport const AffonError = globalThis.AffonError;\nexport default AffonError;\n" },
    .{ .specifier = "affon:runtime", .source = "import 'affon:errors';\n" ++ "export const name = \"affon\";\n" ++ "export const version = \"" ++ version ++ "\";\n" },
    .{ .specifier = "affon:compute", .source = @embedFile("js/compute/public.ts") },
    .{ .specifier = "affon:_internal/compute/program", .source = @embedFile("js/compute/program.ts") },
    .{ .specifier = "affon:ops", .source = @embedFile("js/ops.ts") },
    .{ .specifier = "affon:optim", .source = @embedFile("js/optim.ts") },
    .{ .specifier = "affon:compute/persistence.ts", .source = @embedFile("js/compute/persistence.ts") },
    .{ .specifier = "affon:checkpoint", .source = @embedFile("js/checkpoint/index.ts") },
    .{ .specifier = "affon:dataset", .source = @embedFile("js/dataset/index.ts") },
    .{ .specifier = "affon:dataset/tabular.ts", .source = @embedFile("js/dataset/tabular.ts") },
    .{ .specifier = "affon:dataset/text.ts", .source = @embedFile("js/dataset/text.ts") },
    .{ .specifier = "affon:dataset/text_io.ts", .source = @embedFile("js/dataset/text_io.ts") },
    .{ .specifier = "affon:dataset/text_loader.ts", .source = @embedFile("js/dataset/text_loader.ts") },
    .{ .specifier = "affon:dataset/text_utils.ts", .source = @embedFile("js/dataset/text_utils.ts") },
    .{ .specifier = "affon:dataset/token_windows.ts", .source = @embedFile("js/dataset/token_windows.ts") },
    .{ .specifier = "affon:dataset/tokenizer.ts", .source = @embedFile("js/dataset/tokenizer.ts") },
};

const native_modules = [_]hao.NativeModule{
    .{ .specifier = compute_native.specifier, .load = compute_native.load },
    .{ .specifier = dataset_native.specifier, .load = dataset_native.load },
};

fn installGlobals(context: *packages.PackageContext) !void {
    const value = qjs.eval(context.runtime.ctx, @embedFile("js/errors-global.js"), "<affon:global>", qjs.EvalFlags.global);
    defer qjs.freeValue(context.runtime.ctx, value);
    if (qjs.isException(value)) return error.JavaScriptError;
}

fn packageDefinition() hao.Package { return .{ .name = package_name, .sources = &sources, .native_modules = &native_modules, .install = installGlobals }; }
pub fn registerPackage(registry: *hao.package.Registry) !void { try registry.register(packageDefinition()); }
pub fn register(environment: *hao.RuntimeEnvironment) !void { try environment.registerPackage(packageDefinition()); }

test "registers the candidate-backed Affon package" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
}
