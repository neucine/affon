const std = @import("std");
const hao = @import("hao");
const config = @import("config.zig");
const compute_native = @import("js/compute/native.zig");
const dataset_native = @import("js/dataset/native.zig");
const qjs = hao.qjs;
const packages = hao.package;

pub const package_name = "affon";
pub const version = "0.2.0";
pub const compute = @import("compute");

comptime {
    _ = config;
}
const compute_backend = compute.pipeline.backend;
const compute_compose = compute.pipeline.compose;
const autograd_types = compute.shared.types.autograd;
const autograd_execution = compute.pipeline.execution.autograd;

const sources = [_]hao.SourceModule{
    .{
        .specifier = "affon:runtime",
        .source =
        "export const name = \"affon\";\n" ++
        "export const version = \"" ++ version ++ "\";\n",
    },
    .{
        .specifier = "affon:compute",
        .source = @embedFile("js/compute/index.ts"),
    },
    .{
        .specifier = "affon:compute/graph.ts",
        .source = @embedFile("js/compute/graph.ts"),
    },
    .{
        .specifier = "affon:compute/captured/shadow_metadata.ts",
        .source = @embedFile("js/compute/captured/shadow_metadata.ts"),
    },
    .{
        .specifier = "affon:compute/compile.ts",
        .source = @embedFile("js/compute/compile.ts"),
    },
    .{
        .specifier = "affon:compute/persistence.ts",
        .source = @embedFile("js/compute/persistence.ts"),
    },
    .{
        .specifier = "affon:nn",
        .source = @embedFile("js/nn/index.ts"),
    },
    .{
        .specifier = "affon:checkpoint",
        .source = @embedFile("js/checkpoint/index.ts"),
    },
    .{
        .specifier = "affon:dataset",
        .source = @embedFile("js/dataset/index.ts"),
    },
    .{
        .specifier = "affon:dataset/tabular.ts",
        .source = @embedFile("js/dataset/tabular.ts"),
    },
    .{
        .specifier = "affon:dataset/text.ts",
        .source = @embedFile("js/dataset/text.ts"),
    },
    .{
        .specifier = "affon:dataset/text_io.ts",
        .source = @embedFile("js/dataset/text_io.ts"),
    },
    .{
        .specifier = "affon:dataset/text_loader.ts",
        .source = @embedFile("js/dataset/text_loader.ts"),
    },
    .{
        .specifier = "affon:dataset/text_utils.ts",
        .source = @embedFile("js/dataset/text_utils.ts"),
    },
    .{
        .specifier = "affon:dataset/token_windows.ts",
        .source = @embedFile("js/dataset/token_windows.ts"),
    },
    .{
        .specifier = "affon:dataset/tokenizer.ts",
        .source = @embedFile("js/dataset/tokenizer.ts"),
    },
};

const native_modules = [_]hao.NativeModule{ .{
    .specifier = compute_native.specifier,
    .load = compute_native.load,
}, .{
    .specifier = dataset_native.specifier,
    .load = dataset_native.load,
} };

fn installGlobals(context: *packages.PackageContext) !void {
    const source =
        \\if (typeof globalThis.AffonError !== 'function') {
        \\  globalThis.AffonError = class AffonError extends Error {
        \\    constructor(code, message) {
        \\      super(message);
        \\      this.name = 'AffonError';
        \\      this.code = code;
        \\    }
        \\  };
        \\}
    ;
    const value = qjs.eval(context.runtime.ctx, source, "<affon:global>", qjs.EvalFlags.global);
    defer qjs.freeValue(context.runtime.ctx, value);
    if (qjs.isException(value)) return error.JavaScriptError;
}

fn packageDefinition() hao.Package {
    return .{
        .name = package_name,
        .sources = &sources,
        .native_modules = &native_modules,
        .install = installGlobals,
    };
}

pub fn registerPackage(registry: *hao.package.Registry) !void {
    try registry.register(packageDefinition());
}

pub fn register(environment: *hao.RuntimeEnvironment) !void {
    try environment.registerPackage(packageDefinition());
}

test "registers the Affon package through Hao" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();

    try register(&environment);
    try environment.evalModuleSource(
        "import { name } from 'affon:runtime'; globalThis.__affon_name = name; globalThis.__affon_error = new AffonError('invalid_arg', 'test');",
        "<affon-test>",
    );

    const global = hao.qjs.c.JS_GetGlobalObject(environment.runtime.ctx);
    defer hao.qjs.freeValue(environment.runtime.ctx, global);
    const value = hao.qjs.getProperty(environment.runtime.ctx, global, "__affon_name");
    defer hao.qjs.freeValue(environment.runtime.ctx, value);
    const name = try hao.qjs.valueToStringAlloc(environment.runtime.ctx, value, std.testing.allocator);
    defer std.testing.allocator.free(name);
    try std.testing.expectEqualStrings("affon", name);

    const error_value = hao.qjs.getProperty(environment.runtime.ctx, global, "__affon_error");
    defer hao.qjs.freeValue(environment.runtime.ctx, error_value);
    const error_name = hao.qjs.getProperty(environment.runtime.ctx, error_value, "name");
    defer hao.qjs.freeValue(environment.runtime.ctx, error_name);
    const error_name_text = try hao.qjs.valueToStringAlloc(environment.runtime.ctx, error_name, std.testing.allocator);
    defer std.testing.allocator.free(error_name_text);
    try std.testing.expectEqualStrings("AffonError", error_name_text);
}

test "compute source binding uses Hao's source ABI" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();

    try register(&environment);
    try environment.evalModuleSource(
        \\import { tensor, empty, zeros, ones, full, parameter, copy, grad, compile, module, sgd, adam, adamw, rand, randn, seed, arange, linspace, add, sub, mul, div, matmul, dot, square, gt_scalar, cast, abs, exp, log, neg, sqrt, sign, relu, sigmoid, silu, tanh, gelu, clamp, softmax, sum, mean, min, max, variance, std, argmin, argmax, reshape, slice, at, all, range, Duration, schedules, contiguous, permute, transpose, squeeze, unsqueeze, cat, stack, one_hot, gather, index_select, topk } from 'affon:compute';
        \\(() => {
        \\  const lhs = tensor([1, -2, 3]);
        \\  const rhs = tensor([4, 5, 6]);
        \\  const added = add(lhs, rhs);
        \\  const result = relu(added);
        \\  const difference = sub(rhs, lhs);
        \\  const product = mul(lhs, rhs);
        \\  const quotient = div(rhs, lhs);
        \\  const matrix = tensor([[1, 2], [3, 4]]);
        \\  const created = [empty([2, 2]), zeros([2, 2]), ones([2, 2]), full([2, 2], 3)];
        \\  seed(7);
        \\  const generated = [rand([2]), randn([2]), arange(3), arange(1, 4), linspace(0, 1, 3)];
        \\  const reductions = [dot(lhs, rhs), clamp(lhs, 0, 2), softmax(lhs, 0), sum(lhs), mean(lhs), min(lhs), max(lhs), variance(lhs), std(lhs), argmin(lhs), argmax(lhs), reshape(lhs, [3]), contiguous(lhs)];
        \\  const indices = tensor([1, 0], { dtype: 'i64' });
        \\  const shaped = [permute(matrix, [1, 0]), transpose(matrix, 0, 1), squeeze(unsqueeze(lhs, 0), 0), cat([lhs, rhs]), stack([lhs, rhs]), one_hot(indices, 3), gather(matrix, 0, tensor([[0, 1], [1, 0]], { dtype: 'i64' })), index_select(matrix, 0, indices), slice(matrix, range(0, 2), all)];
        \\  const top = topk(matrix, 1, 1);
        \\  const unary = [abs(lhs), exp(lhs), log(rhs), neg(lhs), sqrt(rhs), sign(lhs), relu(lhs), sigmoid(lhs), silu(lhs), tanh(lhs), gelu(lhs)];
        \\  const scalar = sum(lhs);
        \\  const greater = gt_scalar(lhs, 0);
        \\  const copied = copy(lhs, tensor([8, 9, 10]));
        \\  const linear = schedules.linear({ start: 1, end: 3, duration: Duration.steps(4) });
        \\  const compiled = compile((value) => add(value, value));
        \\  const compiledResult = compiled(tensor([1, 2]));
        \\  const model = module({ weight: parameter([1]) }, (state, value) => mul(value, state.weight));
        \\  const modelResult = model(tensor([3]));
        \\  const modelParameterCount = model.parameters.length;
        \\  const sequence = schedules.sequence(linear, schedules.constant(3));
        \\  const trainable = parameter([1]);
        \\  grad(sum(mul(trainable, trainable)), [trainable]);
        \\  sgd({ lr: 0.01 })([trainable]);
        \\  adam({ lr: 0.001 })([trainable]);
        \\  adamw({ lr: 0.001 })([trainable]);
        \\  void matmul;
        \\  if (result.shape[0] !== 3 || modelResult.shape[0] !== 1 || modelParameterCount !== 1 || compiledResult.to_array()[1] !== 4 || copied.to_array()[0] !== 8 || linear({ epoch: 0, step: 2 }) !== 2 || sequence({ epoch: 0, step: 5 }) !== 3 || trainable.grad === null || greater.shape[0] !== 3 || matrix.shape[0] !== 2 || matrix.shape[1] !== 2 || typeof matrix.item !== 'function' || matrix.to_array()[1][0] !== 3 || !matrix.toString().includes('Tensor') || indices.dtype !== 'i64' || created.length !== 4 || generated.length !== 5 || shaped.length !== 9 || shaped[8].shape[0] !== 2 || top.values.shape[0] !== 2 || top.indices.shape[0] !== 2 || difference.ndim !== 1 || product.ndim !== 1 || quotient.ndim !== 1 || unary.length !== 11 || reductions.length !== 13 || scalar.shape.length !== 1 || scalar.shape[0] !== 1) throw new Error('invalid compute binding result');
        \\})();
    ,
        "<compute-binding-test>",
    );
}

test "native modules keep one identity across package submodules" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();

    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor } from 'affon:compute';
        \\const left = tensor([1, 2]);
        \\const right = tensor([3, 4]);
        \\if (typeof graphSupport.graph !== 'function' || left.shape[0] !== 2 || right.shape[0] !== 2) throw new Error('invalid native module identity');
    , "<native-module-identity-test>");
}

test "compute modules preserve nested state and compiled behavior" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import { tensor, parameter, module, compile } from 'affon:compute';
        \\const child = module({ weight: parameter([1]) }, (state, value) => state.weight);
        \\const parent = module({ child }, (state, value) => state.child(value));
        \\parent.metadata('root');
        \\const saved = parent.state();
        \\parent.eval();
        \\const evalResult = parent(tensor([3]));
        \\parent.train();
        \\parent.restore(saved);
        \\const compiled = compile(parent);
        \\const compiledResult = compiled(tensor([4]));
        \\if (parent.parameters.length !== 1 || parent.training !== true || child.training !== true || child.module_path !== 'root.child' || evalResult.shape[0] !== 1 || compiledResult.shape[0] !== 1 || compiled.parameters.length !== 1) throw new Error('invalid compute module behavior');
    , "<compute-module-parity-test>");
}

test "compute graph captures binary elementwise operations" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, add, mul } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => mul(add(left, right), right));
        \\const result = program.run(tensor([1, 2]), tensor([3, 4]));
        \\if (result.to_array()[0] !== 12 || result.to_array()[1] !== 24) throw new Error('invalid captured binary result');
    , "<compute-graph-binary-test>");
}

test "compute graph captures scalar comparison" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, gt_scalar } from 'affon:compute';
        \\const program = graphSupport.graph((value) => gt_scalar(value, 1));
        \\const result = program.run(tensor([-1, 2, 3]));
        \\const values = result.to_array();
        \\if (values[0] !== 0 || values[1] !== 1 || values[2] !== 1) throw new Error('invalid captured scalar comparison result');
    , "<compute-graph-comparison-test>");
}

test "compute graph captures unary elementwise operations" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, add, relu, exp } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => exp(relu(add(left, right))));
        \\const result = program.run(tensor([-2, 0]), tensor([1, 2]));
        \\const values = result.to_array();
        \\if (Math.abs(values[0] - 1) > 0.0001 || Math.abs(values[1] - Math.exp(2)) > 0.0001) throw new Error('invalid captured unary result');
    , "<compute-graph-unary-test>");
}

test "compute graph captures mean reductions" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, mean } from 'affon:compute';
        \\const program = graphSupport.graph((value) => mean(value, 1));
        \\const result = program.run(tensor([[1, 2], [3, 5]]));
        \\const values = result.to_array();
        \\if (Math.abs(values[0] - 1.5) > 0.0001 || Math.abs(values[1] - 4) > 0.0001) throw new Error('invalid captured mean result');
    , "<compute-graph-reduction-test>");
}

test "compute graph captures variance reductions" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, variance } from 'affon:compute';
        \\const program = graphSupport.graph((value) => variance(value));
        \\const result = program.run(tensor([1, 2, 3, 4]));
        \\if (Math.abs(result.to_array() - 1.25) > 0.0001) throw new Error('invalid captured variance result');
    , "<compute-graph-variance-test>");
}

test "compute graph captures matrix operations" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, matmul, dot } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => matmul(left, right));
        \\const result = program.run(tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]));
        \\const values = result.to_array();
        \\if (values[0][0] !== 19 || values[0][1] !== 22 || values[1][0] !== 43 || values[1][1] !== 50) throw new Error('invalid captured matmul result');
    , "<compute-graph-matrix-test>");
}

test "compute graph captures dot operations" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, dot } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => dot(left, right));
        \\const result = program.run(tensor([1, 2]), tensor([3, 4]));
        \\if (result.to_array() !== 11) throw new Error('invalid captured dot result');
    , "<compute-graph-dot-test>");
}

test "compute graph captures softmax" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, softmax } from 'affon:compute';
        \\const program = graphSupport.graph((value) => softmax(value, 0));
        \\const result = program.run(tensor([0, 1]));
        \\const values = result.to_array();
        \\if (Math.abs(values[0] - 0.2689414) > 0.0001 || Math.abs(values[1] - 0.7310586) > 0.0001) throw new Error('invalid captured softmax result');
    , "<compute-graph-softmax-test>");
}

test "compute graph captures indexed cross entropy" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, cross_entropy_indexed } from 'affon:compute';
        \\const program = graphSupport.graph((logits, targets) => cross_entropy_indexed(logits, targets));
        \\const result = program.run(tensor([[1, 2, 3], [3, 2, 1]]), tensor([2, 0], { dtype: 'i64' }));
        \\if (Math.abs(result.to_array() - 0.4076059) > 0.0001) throw new Error('invalid captured cross entropy result');
    , "<compute-graph-cross-entropy-test>");
}

test "compute graph captures masked fill" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, masked_fill } from 'affon:compute';
        \\const program = graphSupport.graph((value, mask) => masked_fill(value, mask, -9));
        \\const result = program.run(tensor([1, 2]), tensor([0, 1], { dtype: 'i64' }));
        \\const values = result.to_array();
        \\if (values[0] !== 1 || values[1] !== -9) throw new Error('invalid captured masked fill result');
    , "<compute-graph-masked-fill-test>");
}

test "compute graph captures index select" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, index_select } from 'affon:compute';
        \\const program = graphSupport.graph((value, index) => index_select(value, 0, index));
        \\const result = program.run(tensor([[1, 2], [3, 4], [5, 6]]), tensor([2, 0], { dtype: 'i64' }));
        \\const values = result.to_array();
        \\if (values[0][0] !== 5 || values[0][1] !== 6 || values[1][0] !== 1 || values[1][1] !== 2) throw new Error('invalid captured index select result');
    , "<compute-graph-index-select-test>");
}

test "compute graph captures cast" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, cast } from 'affon:compute';
        \\const program = graphSupport.graph((value) => cast(value, 'f64'));
        \\const result = program.run(tensor([1, 2]));
        \\if (result.dtype !== 'f64' || result.to_array()[1] !== 2) throw new Error('invalid captured cast result');
    , "<compute-graph-cast-test>");
}

test "compute graph captures one hot" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, one_hot } from 'affon:compute';
        \\const program = graphSupport.graph((value) => one_hot(value, 3));
        \\const result = program.run(tensor([0, 2], { dtype: 'i64' }));
        \\const values = result.to_array();
        \\if (values[0][0] !== 1 || values[0][1] !== 0 || values[0][2] !== 0 || values[1][0] !== 0 || values[1][1] !== 0 || values[1][2] !== 1) throw new Error('invalid captured one hot result');
    , "<compute-graph-one-hot-test>");
}

test "compute graph captures clamp" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, clamp } from 'affon:compute';
        \\const program = graphSupport.graph((value) => clamp(value, 0, 1));
        \\const result = program.run(tensor([-1, 0.5, 2]));
        \\const values = result.to_array();
        \\if (values[0] !== 0 || values[1] !== 0.5 || values[2] !== 1) throw new Error('invalid captured clamp result');
    , "<compute-graph-clamp-test>");
}

test "compute graph captures where" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, where } from 'affon:compute';
        \\const program = graphSupport.graph((condition, onTrue, onFalse) => where(condition, onTrue, onFalse));
        \\const result = program.run(tensor([1, 0], { dtype: 'i64' }), tensor([10, 20]), tensor([30, 40]));
        \\const values = result.to_array();
        \\if (values[0] !== 10 || values[1] !== 40) throw new Error('invalid captured where result');
    , "<compute-graph-where-test>");
}

test "compute graph captures gather" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, gather } from 'affon:compute';
        \\const program = graphSupport.graph((value, index) => gather(value, 1, index));
        \\const result = program.run(tensor([[10, 20, 30], [40, 50, 60]]), tensor([[2, 0], [1, 2]], { dtype: 'i64' }));
        \\const values = result.to_array();
        \\if (values[0][0] !== 30 || values[0][1] !== 10 || values[1][0] !== 50 || values[1][1] !== 60) throw new Error('invalid captured gather result');
    , "<compute-graph-gather-test>");
}

test "compute graph captures cat" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, cat } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => cat([left, right]));
        \\const result = program.run(tensor([1, 2]), tensor([3, 4]));
        \\const values = result.to_array();
        \\if (values[0] !== 1 || values[3] !== 4) throw new Error('invalid captured cat result');
    , "<compute-graph-cat-test>");
}

test "compute graph captures stack" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, stack } from 'affon:compute';
        \\const program = graphSupport.graph((left, right) => stack([left, right]));
        \\const result = program.run(tensor([1, 2]), tensor([3, 4]));
        \\const values = result.to_array();
        \\if (values[0][0] !== 1 || values[1][1] !== 4) throw new Error('invalid captured stack result');
    , "<compute-graph-stack-test>");
}

test "compute graph captures random values" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { rand, randn } from 'affon:compute';
        \\const program = graphSupport.graph(() => rand([2]));
        \\const result = program.run();
        \\if (result.shape[0] !== 2 || result.dtype !== 'f32') throw new Error('invalid captured rand result');
    , "<compute-graph-random-test>");
}

test "compute graph captures reduction family" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, sum, min, max, std, argmin, argmax } from 'affon:compute';
        \\const program = graphSupport.graph((value) => argmax(value, 1));
        \\const result = program.run(tensor([[1, 2], [3, 4]]));
        \\if (result.to_array()[0] !== 1 || result.to_array()[1] !== 1) throw new Error('invalid captured argmax result');
    , "<compute-graph-reduction-family-test>");
}

test "compute graph captures topk secondary output" {
    var environment = try hao.RuntimeEnvironment.init(std.testing.allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, topk } from 'affon:compute';
        \\const program = graphSupport.graph((value) => topk(value, 2, 1).indices);
        \\const result = program.run(tensor([[1, 4, 2], [9, 3, 7]]));
        \\if (result.shape[0] !== 2 || result.shape[1] !== 2 || result.dtype !== 'i64') throw new Error('invalid captured topk indices');
    , "<compute-graph-topk-secondary-test>");
}

test "compute graph captures reshape" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, reshape } from 'affon:compute';
        \\const program = graphSupport.graph((value) => reshape(value, [2, 2]));
        \\const result = program.run(tensor([1, 2, 3, 4]));
        \\const values = result.to_array();
        \\if (values[0][0] !== 1 || values[0][1] !== 2 || values[1][0] !== 3 || values[1][1] !== 4) throw new Error('invalid captured reshape result');
    , "<compute-graph-reshape-test>");
}

test "compute graph captures shape views" {
    var environment = try hao.RuntimeEnvironment.init(std.heap.page_allocator, .{ .std = false });
    defer environment.deinit();
    try register(&environment);
    try environment.evalModuleSource(
        \\import graphSupport from 'affon:compute/graph.ts';
        \\import { tensor, contiguous, squeeze, unsqueeze, permute } from 'affon:compute';
        \\const program = graphSupport.graph((value) => contiguous(permute(value, [1, 0])));
        \\const result = program.run(tensor([[1, 2], [3, 4]]));
        \\const values = result.to_array();
        \\if (values[0][0] !== 1 || values[0][1] !== 3 || values[1][0] !== 2 || values[1][1] !== 4) throw new Error('invalid captured layout result');
    , "<compute-graph-shape-views-test>");
}

test "compute core owns tensor values independently of the JS binding" {
    const Tensor = compute.types.tensor.Tensor;
    var value = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer value.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, value.shape.dims);
    try std.testing.expectEqual(compute.types.tensor.DType.f32, value.dtype);
    const bytes = try value.storage.?.readableBytes();
    const values = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqual(@as(f32, 3), values[2]);
}

test "compute engine casts tensor dtype" {
    const Tensor = compute.types.tensor.Tensor;
    const value = try Tensor.fromSliceF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer value.deinit();
    const converted = try compute.Engine.init(std.testing.allocator, .{}).cast(value, .f64);
    defer converted.deinit();
    try std.testing.expectEqual(compute.types.tensor.DType.f64, converted.dtype);
}

test "compute CPU kernels consume tensor storage" {
    const Tensor = compute.types.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();
    const result = try Tensor.createContiguous(std.testing.allocator, &.{3}, .f32, .cpu, false);
    defer result.deinit();

    try compute_backend.cpu.add(.f32, lhs.storage.?, rhs.storage.?, result.storage.?);
    const bytes = try result.storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute eager execution runs an operation" {
    const Tensor = compute.types.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer rhs.deinit();

    const inputs = [_]*Tensor{ lhs, rhs };
    const op = try compute.operation.Op.init(.add, &inputs, .{ .none = {} });
    var execution = try compute.Engine.init(std.testing.allocator, .{}).executeRaw(op);
    defer execution.deinit();
    const result = execution.primary;

    const bytes = try result.storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute graph execution runs multiple operations" {
    const Tensor = compute.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ -1, 2, 3 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{3}, &.{ 2, 3, 4 });
    defer rhs.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, rhs);
    const spec = try lhs.spec();
    const sum_id = try graph.addOp(.add, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.relu, &.{sum_id}, .{ .none = {} }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.Engine.init(std.testing.allocator, .{}).executeGraph(&graph, &.{ lhs, rhs });
    defer result.deinit();
    const bytes = try result.outputs[0].storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 1, 5, 7 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute graph execution fuses matmul and full-shape bias" {
    const Tensor = compute.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer rhs.deinit();
    const bias = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer bias.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, rhs);
    const bias_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, bias);
    var output_shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer output_shape.deinit();
    var output_layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, output_shape);
    defer output_layout.deinit();
    const output_spec = compute.types.tensor.TensorSpec{
        .shape = output_shape,
        .dtype = .f32,
        .layout = output_layout,
        .device = .cpu,
    };
    const matmul_id = try graph.addOp(.matmul, &.{ lhs_id, rhs_id }, .{ .none = {} }, output_spec);
    const output_id = try graph.addOp(.add, &.{ matmul_id, bias_id }, .{ .none = {} }, output_spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.Engine.init(std.testing.allocator, .{}).executeGraph(&graph, &.{ lhs, rhs, bias });
    defer result.deinit();
    const bytes = try result.outputs[0].storage.?.readableBytes();
    try std.testing.expectEqualSlices(f32, &.{ 23, 29, 50, 65 }, std.mem.bytesAsSlice(f32, bytes));
}

test "compute graph execution supports matmul add gelu epilogues" {
    const Tensor = compute.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer rhs.deinit();
    const bias = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer bias.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, rhs);
    const bias_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, bias);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.TensorSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const matmul_id = try graph.addOp(.matmul, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const add_id = try graph.addOp(.add, &.{ matmul_id, bias_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.gelu, &.{add_id}, .{ .none = {} }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.Engine.init(std.testing.allocator, .{}).executeGraph(&graph, &.{ lhs, rhs, bias });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectEqual(@as(usize, 4), values.len);
    for (values) |value| try std.testing.expect(value > 0);
}

test "compute graph execution supports add layer norm fusion" {
    const Tensor = compute.tensor.Tensor;
    const lhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 3, 5, 7 });
    defer lhs.deinit();
    const rhs = try Tensor.fromSliceF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 1, 1, 1 });
    defer rhs.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const lhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, lhs);
    const rhs_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, rhs);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.TensorSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const sum_id = try graph.addOp(.add, &.{ lhs_id, rhs_id }, .{ .none = {} }, spec);
    const output_id = try graph.addOp(.layer_norm, &.{sum_id}, .{ .layer_norm = .{ .axis = 1, .eps = 1e-5 } }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.Engine.init(std.testing.allocator, .{}).executeGraph(&graph, &.{ lhs, rhs });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 0), values[0] + values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), values[2] + values[3], 1e-5);
}

test "compute graph execution supports attention score fusion" {
    const Tensor = compute.tensor.Tensor;
    const q = try Tensor.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 0, 0, 1 });
    defer q.deinit();
    const k_t = try Tensor.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 0, 0, 1 });
    defer k_t.deinit();
    const scale = try Tensor.fromSliceF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer scale.deinit();
    const mask = try Tensor.fromSliceI64(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 0, 0, 0, 0 });
    defer mask.deinit();

    var graph = compute.types.ir.Graph.init(std.testing.allocator);
    defer graph.deinit();
    const q_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, q);
    const k_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, k_t);
    const scale_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, scale);
    const mask_id = try compute_compose.builder.addInputFromTensor(std.testing.allocator, &graph, mask);
    var shape = try compute.types.tensor.Shape.initCopy(std.testing.allocator, &.{ 1, 2, 2 });
    defer shape.deinit();
    var layout = try compute.types.tensor.Layout.initContiguous(std.testing.allocator, shape);
    defer layout.deinit();
    const spec = compute.types.tensor.TensorSpec{ .shape = shape, .dtype = .f32, .layout = layout, .device = .cpu };
    const mm_id = try graph.addOp(.matmul, &.{ q_id, k_id }, .{ .none = {} }, spec);
    const scaled_id = try graph.addOp(.mul, &.{ mm_id, scale_id }, .{ .none = {} }, spec);
    const masked_id = try graph.addOp(.masked_fill, &.{ scaled_id, mask_id }, .{ .masked_fill = .{ .value = -1e9 } }, spec);
    const output_id = try graph.addOp(.softmax, &.{masked_id}, .{ .softmax = .{ .axis = 2 } }, spec);
    try graph.setOutputs(&.{output_id});

    var result = try compute.Engine.init(std.testing.allocator, .{}).executeGraph(&graph, &.{ q, k_t, scale, mask });
    defer result.deinit();
    const values = std.mem.bytesAsSlice(f32, try result.outputs[0].storage.?.readableBytes());
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[0] + values[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), values[2] + values[3], 1e-5);
}

test "autograd tape tracks provenance without coupling to execution" {
    const Tensor = compute.tensor.Tensor;
    const input = try Tensor.fromSliceF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    const output = try Tensor.fromSliceF32(std.testing.allocator, &.{2}, &.{ 3, 4 });

    _ = try autograd_types.State.create(std.testing.allocator, input, true);
    _ = try autograd_types.State.create(std.testing.allocator, output, false);
    const node = try autograd_execution.createNode(
        std.testing.allocator,
        .relu,
        &.{.{ .value = input, .input_slot = 0 }},
        &.{input},
        output,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
    );
    autograd_types.State.fromTensor(output).?.attachNode(node);

    try std.testing.expectEqual(autograd_types.TrackingState.trainable, autograd_types.State.trackingState(input));
    try std.testing.expectEqual(autograd_types.TrackingState.tracked, autograd_types.State.trackingState(output));
    autograd_execution.releaseOwnedTensor(output);
    autograd_execution.releaseOwnedTensor(input);
}

test "autograd derives and executes a graph from provenance" {
    const Tensor = compute.tensor.Tensor;
    const allocator = std.testing.allocator;
    const x = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 2, 4 });
    const y = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 1, 3 });
    const sum = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 3, 7 });
    const loss = try Tensor.fromSliceF32(allocator, &.{}, &.{5});

    _ = try autograd_types.State.create(allocator, x, true);
    _ = try autograd_types.State.create(allocator, y, true);
    _ = try autograd_types.State.create(allocator, sum, false);
    _ = try autograd_types.State.create(allocator, loss, true);

    const sum_node = try autograd_execution.createNode(
        allocator,
        .add,
        &.{ .{ .value = x, .input_slot = 0 }, .{ .value = y, .input_slot = 1 } },
        &.{ x, y },
        sum,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
    );
    autograd_types.State.fromTensor(sum).?.attachNode(sum_node);

    const loss_node = try autograd_execution.createNode(
        allocator,
        .mean_all,
        &.{.{ .value = sum, .input_slot = 0 }},
        &.{sum},
        loss,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
    );
    autograd_types.State.fromTensor(loss).?.attachNode(loss_node);

    var derived = try compute_compose.derive.buildFromLossTensor(loss);
    defer derived.deinit(allocator);
    try compute_compose.derive.executeForTensor(loss, &derived);

    const x_state = autograd_types.State.fromTensor(x).?;
    const y_state = autograd_types.State.fromTensor(y).?;
    try std.testing.expect(x_state.gradient != null);
    try std.testing.expect(y_state.gradient != null);
    const x_grad = std.mem.bytesAsSlice(f32, try x_state.gradient.?.storage.?.readableBytes());
    const y_grad = std.mem.bytesAsSlice(f32, try y_state.gradient.?.storage.?.readableBytes());
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, x_grad);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5 }, y_grad);

    autograd_execution.releaseOwnedTensor(loss);
    autograd_execution.releaseOwnedTensor(sum);
    autograd_execution.releaseOwnedTensor(y);
    autograd_execution.releaseOwnedTensor(x);
}
