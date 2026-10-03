import { add, cat, contiguous, cross_entropy, embedding, index_select, layer_norm, masked_fill, mean, mul, reshape, slice, squeeze, unsqueeze } from 'affon:ops'
import { describe, expect, test } from "std:test"
import {
  Session,
  Tensor,
  gradient,
  losses,
  optimize,
  program,
} from "affon:compute"
import { accumulate, adam, adamw, scheduled, schedules, sgd } from "affon:optim"
import * as compute from "affon:compute"
import * as optim from "affon:optim"

describe("compute Program API", () => {
  test("exports only constructible public runtime entry points", () => {
    expect((compute as any).default).toBe(undefined)
    expect((optim as any).default).toBe(undefined)
    expect((compute as any).FormalTensor).toBe(undefined)
    expect((compute as any).ProgramBuilder).toBe(undefined)
    expect((compute as any).Executable).toBe(undefined)
    expect((compute as any).ExecutionState).toBe(undefined)
  })

  test("keeps formal and evaluated dtype and axis contracts aligned", () => {
    expect(() => Tensor.spec("bool" as any, [2])).toThrow("unsupported TensorSpec dtype")
    expect(() => Tensor.f32([Number.MAX_SAFE_INTEGER, 2])).toThrow("element count")
    const session = new Session({ device: "cpu" })
    expect(() => session.tensor([[1, 2]], { axes: ["feature"] })).toThrow("one non-empty name per dimension")
    const indices = session.tensor([0, 1], { dtype: "i64", axes: ["token"] })
    expect(indices.dtype).toBe("i64")
    expect(indices.axes).toEqual(["token"])
    indices.dispose()
    session.dispose()
  })

  test("validates and snapshots parameter initializers during authoring", () => {
    expect(() => program("bad_uniform", p => p.parameter("weight", Tensor.f32([2]), {
      initializer: { kind: "uniform", min: 2, max: 1 },
    }))).toThrow("min <= max")
    expect(() => program("bad_normal", p => p.parameter("weight", Tensor.f32([2]), {
      initializer: { kind: "normal", standard_deviation: -1 },
    }))).toThrow("standard_deviation")
    expect(() => program("bad_xavier", p => p.parameter("weight", Tensor.f32([0]), {
      initializer: { kind: "xavier_uniform" },
    }))).toThrow("fan-in")
    expect(() => program("unknown_initializer", p => p.parameter("weight", Tensor.f32([2]), {
      initializer: { kind: "custom" } as any,
    }))).toThrow("initializer kind")

    const mutable = { kind: "normal" as const, mean: 2, standard_deviation: 0.5 }
    const source = program("initializer_snapshot", p => p.parameter("weight", Tensor.f32([2]), { initializer: mutable }))
    mutable.mean = 99
    const parameter = source.inspect().nodes.find(node => node.role === "parameter")!
    expect((parameter.options as any).initializer).toEqual({ kind: "normal", mean: 2, standard_deviation: 0.5 })
    expect(Object.isFrozen((parameter.options as any).initializer)).toBe(true)
  })

  test("validates and snapshots constants and Session tensor data", () => {
    expect(() => program("wrong_constant_shape", p => p.constant("value", [1, 2], Tensor.f32([1])))).toThrow("TensorSpec")
    expect(() => program("ragged_constant", p => p.constant("value", [[1], [2, 3]] as any, Tensor.f32([2, 2])))).toThrow("rectangular")
    expect(() => program("nonfinite_constant", p => p.constant("value", Number.NaN, Tensor.f32([1])))).toThrow("finite")
    expect(() => program("fractional_integer_constant", p => p.constant("value", [1.5], Tensor.i64([1])))).toThrow("safe integers")

    const values = [[1, 2], [3, 4]]
    const source = program("constant_snapshot", p => p.constant("value", values, Tensor.f32([2, 2])))
    values[0][0] = 99
    const captured = source.inspect().nodes.find(node => node.role === "constant")!.value
    expect(captured).toEqual([[1, 2], [3, 4]])
    expect(Object.isFrozen(captured)).toBe(true)
    expect(Object.isFrozen((captured as any)[0])).toBe(true)

    const session = new Session()
    expect(() => session.tensor([[1], [2, 3]] as any)).toThrow("rectangular")
    expect(() => session.tensor([1.5], { dtype: "i64" })).toThrow("safe integers")
    expect(() => session.tensor([1], { dtype: "bool" as any })).toThrow("unsupported Tensor dtype")
    session.dispose()
  })

  test("supports optional deterministic disposal and safe child lifetimes", () => {
    const identity = program("lifetime_identity", p => p.argument("value", Tensor.f32([2])))
    const session = new Session()
    const value = session.tensor([1, 2])

    expect(value.disposed).toBe(false)
    expect(session.owns(value)).toBe(true)
    value[Symbol.dispose]()
    expect(value.disposed).toBe(true)
    expect(session.owns(value)).toBe(false)
    expect(() => session.compile(identity).run({ value })).toThrow("not a Tensor owned by this Session")

    const live = session.tensor([3, 4])
    session.dispose()
    expect(session.owns(live)).toBe(false)
    expect(() => add(live, live)).toThrow("Session has been disposed")
    expect(live.to_array()).toEqual([3, 4])
    live.dispose()
    expect(live.disposed).toBe(true)
  })

  test("recompiles Programs after a cached Executable is disposed", () => {
    const source = program("recompile_identity", p => p.argument("value", Tensor.f32([2])))
    const session = new Session()
    const first = session.compile(source)
    first.dispose()

    expect(first.disposed).toBe(true)
    const rejected = session.tensor([1, 2])
    expect(() => first.run({ value: rejected })).toThrow("Executable has been disposed")
    rejected.dispose()

    const second = session.compile(source)
    const value = session.tensor([3, 4])
    const result = second.run({ value })
    expect(second === first).toBe(false)
    expect(second.disposed).toBe(false)
    expect(result.to_array()).toEqual([3, 4])

    result.dispose()
    value.dispose()
    session.dispose()
    expect(second.disposed).toBe(true)
  })

  test("caches compiled executables by Program identity", () => {
    const first = program("same_name", p => {
      const value = p.argument("value", Tensor.f32([2]))
      add(value, value)
      return value
    })
    const second = program("same_name", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return add(value, value)
    })
    expect(first.provenance === second.provenance).toBe(false)

    const session = new Session()
    const value = session.tensor([2, 3])
    const firstExecutable = session.compile(first)
    const secondExecutable = session.compile(second)
    const firstResult = firstExecutable.run({ value })
    const secondResult = secondExecutable.run({ value })

    expect(firstExecutable === secondExecutable).toBe(false)
    expect(firstResult.to_array()).toEqual([2, 3])
    expect(secondResult.to_array()).toEqual([4, 6])

    firstResult.dispose()
    secondResult.dispose()
    value.dispose()
    session.dispose()
  })

  test("authors, composes, transforms, and inspects one symbolic pipeline", () => {
    const imageSpec = Tensor.f32([2, 4], { axes: ["batch", "feature"] })
    const model = program("classifier", p => {
      const image = p.argument("image", imageSpec)
      return p.nn.linear(image, { name: "head", out_features: 3 })
    })
    const composed = program("ensemble", p => {
      const image = p.argument("image", imageSpec)
      const first = model(image)
      const second = p.use(model, { as: "second", image })
      return add(first, second as typeof first)
    })
    const loss = program("ensemble_loss", p => {
      const image = p.argument("image", imageSpec)
      const labels = p.argument("labels", Tensor.i64([2]))
      return cross_entropy(composed(image), labels)
    })
    const optimizationLoss = program("ensemble_optimization_loss", p => cross_entropy(
      p.argument("logits", Tensor.f32([2, 3])),
      p.argument("labels", Tensor.i64([2])),
    ))
    const gradients = gradient(loss, ["ensemble_classifier_head_weight", "ensemble_second_head_weight"])
    const step = optimize(composed, optimizationLoss, adam({ learning_rate: 0.001 }))
    const fasterStep = optimize(composed, optimizationLoss, adam({ learning_rate: 0.01 }))

    expect(loss.inspect().arguments.map(value => [value.name, value.spec.dtype, value.spec.shape])).toEqual([
      ["image", "f32", [2, 4]],
      ["labels", "i64", [2]],
    ])
    expect(loss.inspect().parameters.map(value => value.name)).toEqual([
      "ensemble_classifier_head_weight",
      "ensemble_classifier_head_bias",
      "ensemble_second_head_weight",
      "ensemble_second_head_bias",
    ])
    expect(gradients.inspect().outputs.length).toBe(2)
    expect(step.inspect().transitions[0].kind).toBe("optimize")
    expect(step.provenance === fasterStep.provenance).toBe(false)
  })

  test("enforces exact composition bindings and deeply immutable inspection metadata", () => {
    const child = program("composition_child", p => add(
      p.argument("left", Tensor.f32([2])),
      p.argument("right", Tensor.f32([2])),
    ))
    expect(() => program("extra_positional", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return (child as any)(value, value, value)
    })).toThrow("expects 2 arguments")
    expect(() => program("extra_named", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return p.use(child, { as: "child", left: value, right: value, extra: value })
    })).toThrow("unknown composition_child argument")

    const scalar = program("composition_scalar", p => mean(p.argument("value", Tensor.f32([2]))))
    const derivative = gradient(scalar, "value")
    expect(() => program("composed_gradient", p => derivative(p.argument("value", Tensor.f32([2]))))).toThrow("only authored Programs")

    const shaped = program("immutable_metadata", p => reshape(p.argument("value", Tensor.f32([2, 2])), [4]))
    const options = shaped.inspect().nodes.at(-1)!.options as any
    expect(Object.isFrozen(options)).toBe(true)
    expect(Object.isFrozen(options.shape)).toBe(true)
  })

  test("combines reusable model and loss Programs while preserving inference state", () => {
    const model = program("reusable_classifier", p => {
      const x = p.argument("x", Tensor.f32([2, 2]))
      return p.nn.linear(x, { name: "head", out_features: 2 })
    })
    const loss = program("classification_loss", p => cross_entropy(
      p.argument("logits", Tensor.f32([2, 2])),
      p.argument("labels", Tensor.i64([2])),
    ))
    const train = optimize(model, loss, sgd({ learning_rate: 0.05 }))

    expect(train.inspect().arguments.map(value => value.name)).toEqual(["x", "labels"])
    expect(train.inspect().parameters.map(value => value.provenance)).toEqual(
      model.inspect().parameters.map(value => value.provenance),
    )

    const session = new Session()
    const state = session.initialize(train, { seed: 7 })
    const x = session.tensor([[1, 0], [0, 1]])
    const labels = session.tensor([0, 1], { dtype: "i64" })
    const trainingLoss = session.compile(train).run({ x, labels }, state)
    const logits = session.compile(model).run({ x }, state)

    expect(Number.isFinite(trainingLoss.item())).toBe(true)
    expect(logits.shape).toEqual([2, 2])
    expect(state.optimizer_state.$step).toBe(1)

    trainingLoss.dispose()
    logits.dispose()
    x.dispose()
    labels.dispose()
    state.dispose()
    session.dispose()
  })

  test("materializes built-in loss templates in the losses namespace", () => {
    const model = program("builtin_loss_model", p => p.nn.linear(
      p.argument("x", Tensor.f32([2, 2])),
      { name: "head", out_features: 2 },
    ))
    const optimizer = sgd({ learning_rate: 0.01 })
    const templates = [
      losses.cross_entropy(),
      losses.mean_squared_error(),
      losses.binary_cross_entropy(),
      losses.binary_cross_entropy_with_logits(),
    ]
    expect(templates.every(Object.isFrozen)).toBe(true)
    const steps = templates.map(template => optimize(model, template, optimizer))
    expect(steps[0].inspect().arguments.map(value => [value.name, value.spec.dtype, value.spec.shape])).toEqual([
      ["x", "f32", [2, 2]],
      ["labels", "i64", [2]],
    ])
    for (const step of steps.slice(1)) {
      expect(step.inspect().arguments.map(value => [value.name, value.spec.dtype, value.spec.shape])).toEqual([
        ["x", "f32", [2, 2]],
        ["target", "f32", [2, 2]],
      ])
    }
  })

  test("validates, freezes, and snapshots optimizer descriptors", () => {
    expect(Object.isFrozen(sgd())).toBe(true)
    expect(Object.isFrozen(adam())).toBe(true)
    expect(Object.isFrozen(adamw())).toBe(true)
    expect(Object.isFrozen(accumulate(adam(), { steps: 4 }))).toBe(true)
    expect(Object.isFrozen(scheduled(adam(), schedules.cosine({ start: 0.1, end: 0.01, steps: 10 })))).toBe(true)
    expect(() => accumulate(adam(), { steps: 1 })).toThrow("at least 2")
    expect(() => schedules.linear({ start: 0.1, end: 0, steps: 0 })).toThrow("positive integer")
    expect(() => schedules.warmupCosine({ start: 0, peak: 0.1, end: 0, warmup_steps: 10, total_steps: 10 })).toThrow("less than")
    expect(() => sgd({ momentum: 1 })).toThrow("momentum")
    expect(() => adam({ epsilon: 0 })).toThrow("epsilon")

    const model = program("optimizer_snapshot_model", p => {
      const value = p.argument("value", Tensor.f32([1]))
      const weight = p.parameter("weight", Tensor.f32([1]))
      return mul(value, weight)
    })
    const loss = program("optimizer_snapshot_loss", p => mean(
      p.argument("prediction", Tensor.f32([1])),
    ))
    const mutable = { kind: "sgd", learning_rate: 0.25, momentum: 0.5 } as any
    const step = optimize(model, loss, mutable)
    mutable.learning_rate = 9
    mutable.momentum = 0.9
    const captured = step.inspect().transitions[0].optimizer
    expect(captured.learning_rate).toBe(0.25)
    expect(captured.kind === "sgd" && captured.momentum).toBe(0.5)
    expect(Object.isFrozen(captured)).toBe(true)
    expect(() => optimize(model, loss, { kind: "custom" } as any)).toThrow("Optimizer kind")
  })

  test("accumulates gradients before applying one optimizer update", () => {
    const model = program("accumulating_model", p => {
      const value = p.argument("value", Tensor.f32([1]))
      const weight = p.parameter("weight", Tensor.f32([1]), { initializer: { kind: "constant", value: 1 } })
      return mul(value, weight)
    })
    const loss = program("accumulating_loss", p => mean(
      p.argument("prediction", Tensor.f32([1])),
    ))
    const train = optimize(model, loss, accumulate(sgd({ learning_rate: 0.1 }), { steps: 2 }))
    const session = new Session()
    const state = session.initialize(train)
    const value = session.tensor([2])
    const parameterKey = train.inspect().parameters[0].provenance
    const initial = state.parameters[parameterKey].to_array()

    const firstLoss = session.compile(train).run({ value }, state)
    expect(state.parameters[parameterKey].to_array()).toEqual(initial)
    expect(state.optimizer_state.$microstep).toBe(1)
    expect(state.optimizer_state.$step).toBe(undefined)

    const secondLoss = session.compile(train).run({ value }, state)
    expect(Math.abs((state.parameters[parameterKey].to_array() as number[])[0] - 0.8) < 1e-5).toBe(true)
    expect(state.optimizer_state.$microstep).toBe(0)
    expect(state.optimizer_state.$step).toBe(1)
    expect(Object.keys(state.optimizer_state).some(key => key.endsWith("/accumulate/gradient"))).toBe(false)

    firstLoss.dispose()
    secondLoss.dispose()
    value.dispose()
    state.dispose()
    session.dispose()
  })

  test("advances schedules on optimizer updates rather than accumulation microsteps", () => {
    const model = program("scheduled_accumulating_model", p => mul(
      p.argument("value", Tensor.f32([1])),
      p.parameter("weight", Tensor.f32([1]), { initializer: { kind: "constant", value: 1 } }),
    ))
    const loss = program("scheduled_accumulating_loss", p => mean(p.argument("prediction", Tensor.f32([1]))))
    const optimizer = accumulate(scheduled(
      sgd(),
      schedules.linear({ start: 0.1, end: 0.2, steps: 1 }),
    ), { steps: 2 })
    const train = optimize(model, loss, optimizer)
    const captured = train.inspect().transitions[0].optimizer
    expect(Object.isFrozen(captured)).toBe(true)
    expect(captured.kind === "accumulate" && captured.optimizer.kind === "scheduled" && captured.optimizer.schedule.kind).toBe("linear")

    const session = new Session()
    const state = session.initialize(train)
    const value = session.tensor([2])
    const parameterKey = train.inspect().parameters[0].provenance
    const executable = session.compile(train)
    const losses = [
      executable.run({ value }, state),
      executable.run({ value }, state),
      executable.run({ value }, state),
    ]
    expect(Math.abs((state.parameters[parameterKey].to_array() as number[])[0] - 0.8) < 1e-5).toBe(true)
    expect(state.optimizer_state.$step).toBe(1)
    losses.push(executable.run({ value }, state))
    expect(Math.abs((state.parameters[parameterKey].to_array() as number[])[0] - 0.4) < 1e-5).toBe(true)
    expect(state.optimizer_state.$step).toBe(2)

    for (const output of losses) output.dispose()
    value.dispose()
    state.dispose()
    session.dispose()
  })

  test("executes, initializes, caches, validates ownership, and optimizes", () => {
    const identity = program("identity", p => p.argument("value", Tensor.f32([2])))
    const firstSession = new Session({ device: "cpu" })
    const secondSession = new Session({ device: "cpu" })
    const executable = firstSession.compile(identity)

    expect(firstSession.compile(identity) === executable).toBe(true)
    expect(executable.argument_names).toEqual(["value"])
    expect(executable.argument_specs.map(formal => formal.spec.shape)).toEqual([[2]])
    expect(executable.native_input_order).toEqual(["value"])
    const value = firstSession.tensor([2, 3])
    const identityResult = executable.run({ value }) as any
    expect(identityResult.to_array()).toEqual([2, 3])
    const foreignValue = secondSession.tensor([2, 3])
    expect(() => executable.run({ value: foreignValue })).toThrow()

    const model = program("trainable_model", p => {
      return p.nn.linear(p.argument("x", Tensor.f32([2, 2])), { name: "head", out_features: 2 })
    })
    const step = optimize(model, losses.cross_entropy(), adam({ learning_rate: 0.01 }))
    const state = firstSession.initialize(step, { seed: 7 })
    const x = firstSession.tensor([[1, 0], [0, 1]])
    const labels = firstSession.tensor([0, 1], { dtype: "i64" })
    const result = firstSession.compile(step).run({ x, labels }, state) as any
    expect(Number.isFinite(result.item())).toBe(true)
    expect(state.optimizer_state.$step).toBe(1)
    const secondResult = firstSession.compile(step).run({ x, labels }, state) as any
    expect(Number.isFinite(secondResult.item())).toBe(true)
    expect(state.optimizer_state.$step).toBe(2)

    result.dispose()
    secondResult.dispose()
    state.dispose()
    identityResult.dispose()
    value.dispose()
    foreignValue.dispose()
    x.dispose()
    labels.dispose()

    firstSession.dispose()
    secondSession.dispose()
  })

  test("copies and validates ExecutionState initializer values", () => {
    const source = program("initialized", p => {
      const scale = p.parameter("scale", Tensor.f32([2]))
      return mul(p.argument("value", Tensor.f32([2])), scale)
    })
    const session = new Session()
    const supplied = session.tensor([2, 3])
    const state = session.initialize(source, { parameters: { scale: supplied } })
    expect(state.parameters["initialized.scale"] === supplied).toBe(false)
    state.dispose()
    expect(supplied.to_array()).toEqual([2, 3])

    expect(() => session.initialize(source, { parameters: { scale: [[1, 2]] } })).toThrow("TensorSpec")
    const wrongDtype = session.tensor([2, 3], { dtype: "f64" })
    expect(() => session.initialize(source, { parameters: { scale: wrongDtype } })).toThrow("TensorSpec")
    expect(wrongDtype.to_array()).toEqual([2, 3])

    wrongDtype.dispose()
    supplied.dispose()
    session.dispose()
  })

  test("validates live ExecutionState tensors before backend execution", () => {
    const source = program("state_validation", p => mul(
      p.argument("value", Tensor.f32([2])),
      p.parameter("scale", Tensor.f32([2])),
    ))
    const session = new Session()
    const foreignSession = new Session()
    const executable = session.compile(source)
    const state = session.initialize(source, { parameters: { scale: [2, 3] } })
    const original = state.parameters["state_validation.scale"]
    const value = session.tensor([1, 1])
    const wrongArgument = session.tensor([1])
    expect(() => executable.run({ value: wrongArgument }, state)).toThrow("TensorSpec")

    const wrongShape = session.tensor([2])
    state.parameters["state_validation.scale"] = wrongShape
    expect(() => executable.run({ value }, state)).toThrow("TensorSpec")
    state.parameters["state_validation.scale"] = original
    wrongShape.dispose()

    const foreign = foreignSession.tensor([2, 3])
    state.parameters["state_validation.scale"] = foreign
    expect(() => executable.run({ value }, state)).toThrow("not a Tensor owned by this Session")
    state.parameters["state_validation.scale"] = original

    const disposed = session.tensor([2, 3])
    disposed.dispose()
    state.parameters["state_validation.scale"] = disposed
    expect(() => executable.run({ value }, state)).toThrow("not a Tensor owned by this Session")
    state.parameters["state_validation.scale"] = original

    delete state.parameters["state_validation.scale"]
    expect(() => executable.run({ value }, state)).toThrow("missing parameter")
    state.parameters["state_validation.scale"] = original

    wrongArgument.dispose()
    foreign.dispose()
    value.dispose()
    state.dispose()
    session.dispose()
    foreignSession.dispose()
  })

  test("lowers shaped constants, model state, axes, and f64 gradients", () => {
    const session = new Session({ device: "cpu" })
    const spec = Tensor.f64([2], { axes: ["feature"] })
    const shifted = program("shifted", p => {
      const value = p.argument("value", spec)
      const offset = p.constant("offset", [2, 4], spec)
      const running = p.state("running", spec, { initializer: { kind: "ones" } })
      return add(add(value, offset), running)
    })
    const state = session.initialize(shifted)
    const value = session.tensor([1, 3], { dtype: "f64", axes: ["feature"] }) as any
    const shiftedResult = session.compile(shifted).run({ value }, state) as any
    expect(shiftedResult.to_array()).toEqual([4, 8])
    expect(shiftedResult.axes).toEqual(["feature"])

    const average = program("average", p => mean(p.argument("value", spec)))
    const derivative = gradient(average, "value")
    const gradientResult = session.compile(derivative).run({ value }) as any
    expect(gradientResult.to_array()).toEqual([0.5, 0.5])

    gradientResult.dispose()
    shiftedResult.dispose()
    value.dispose()
    state.dispose()
    session.dispose()
  })

  test("executes model-oriented indexing, views, masking, concatenation, and normalization", () => {
    const session = new Session({ device: "cpu" })
    const modelOps = program("model_ops", p => {
      const table = p.argument("table", Tensor.f32([3, 2]))
      const indices = p.argument("indices", Tensor.i64([2]))
      const mask = p.argument("mask", Tensor.i64([2, 2]))
      const embedded = embedding(table, indices)
      const selected = index_select(table, 0, indices)
      const column = contiguous(squeeze(unsqueeze(slice(embedded, [{ start: 0, stop: 2 }, { start: 0, stop: 1 }]), 0), 0))
      const joined = cat([column, column], 1)
      const masked = masked_fill(joined, mask, -9)
      return [embedded, selected, masked, layer_norm(masked, -1)]
    })
    const table = session.tensor([[1, 2], [3, 4], [5, 6]])
    const indices = session.tensor([2, 0], { dtype: "i64" })
    const mask = session.tensor([[0, 1], [1, 0]], { dtype: "i64" })
    const outputs = session.compile(modelOps).run({ table, indices, mask }) as any[]

    expect(outputs[0].to_array()).toEqual([[5, 6], [1, 2]])
    expect(outputs[1].to_array()).toEqual([[5, 6], [1, 2]])
    expect(outputs[2].to_array()).toEqual([[5, -9], [-9, 1]])
    expect((outputs[3].to_array() as number[][]).flat().every(Number.isFinite)).toBe(true)

    for (const output of outputs) output.dispose()
    table.dispose()
    indices.dispose()
    mask.dispose()
    session.dispose()
  })
})
