import { add, cross_entropy, mean, mul, sub } from "affon:ops"
import { describe, expect, test } from "std:test"
import telemetry from "std:telemetry"
import { Session, Tensor, gradient, optimize, program, update_parameters } from "affon:compute"
import {
  binary_cross_entropy,
  binary_cross_entropy_with_logits,
  cross_entropy as crossEntropyLoss,
  linear,
  mean_squared_error,
  type LossCallable,
} from "affon:nn"
import { accumulate, adam, adamw, scheduled, schedules, sgd } from "affon:optim"

describe("Program optimization", () => {
  test("controls compile variants and publishes runtime telemetry", () => {
    const source = program("public_optimization_controls", p => add(
      p.argument("left", Tensor.f32([2])),
      p.argument("right", Tensor.f32([2])),
    ))
    const session = new Session({
      device: "cpu",
      determinism: "strict",
      telemetry: { backendTiming: true, hardwareMetrics: true },
    })
    const summary = session.compile(source, { optimizationLevel: "safe", explanationLevel: "summary" })
    const reference = session.compile(source, { optimizationLevel: "none", explanationLevel: "detailed" })
    const optimized = session.compile(source, { optimizationLevel: "safe", explanationLevel: "detailed" })
    expect(reference === optimized).toBe(false)
    expect(summary === optimized).toBe(false)
    expect(session.compile(source, { optimizationLevel: "safe", explanationLevel: "detailed" }) === optimized).toBe(true)
    const summaryExplanation = summary.explain() as any
    const detailedExplanation = optimized.explain() as any
    expect(detailedExplanation.schema_version).toBe(1)
    expect(detailedExplanation.executable.optimizations.length >= summaryExplanation.executable.optimizations.length).toBe(true)

    const left = session.tensor([1, 2])
    const right = session.tensor([3, 4])
    const beforeRuns = telemetry.metrics().find(metric => metric.scope === "compute.execution" && metric.name === "runs_total")?.value ?? 0
    const result = telemetry.trace("test.optimized_add", () => optimized.run({ left, right }))
    expect(result.to_array()).toEqual([4, 6])
    const metrics = telemetry.metrics()
    expect(metrics.find(metric => metric.scope === "compute.execution" && metric.name === "runs_total")?.value).toBe(beforeRuns + 1)
    expect((metrics.find(metric => metric.scope === "compute.backend" && metric.name === "elapsed_nanoseconds_total")?.value ?? 0) >= 0).toBe(true)

    result.dispose()
    left.dispose()
    right.dispose()
    session.dispose()
  })

  test("updates explicitly selected parameters from a gradient Program", () => {
    const objective = program("low_level_objective", p => {
      const value = p.argument("value", Tensor.f32([1]))
      const weight = p.parameter("weight", Tensor.f32([1]), { initializer: { kind: "ones" } })
      const bias = p.parameter("bias", Tensor.f32([1]), { initializer: { kind: "ones" } })
      return mean(add(mul(value, weight), bias))
    })
    const derivatives = gradient(objective, "weight")
    const update = update_parameters(objective, derivatives, sgd({ learning_rate: 0.1 }))

    expect(update.inspect().transitions[0].parameters).toEqual(["low_level_objective.weight"])
    expect(update.inspect().transitions[0].parameter_ids).toEqual([
      update.inspect().nodes.find(node => node.provenance === "low_level_objective.weight")!.id,
    ])
    const session = new Session()
    const state = session.initialize(update)
    const value = session.tensor([2])
    const result = session.compile(update).run({ value }, state)

    expect(result.item()).toBe(3)
    expect(Math.abs((state.parameters["low_level_objective.weight"].to_array() as number[])[0] - 0.8) < 1e-5).toBe(true)
    expect(state.parameters["low_level_objective.bias"].to_array()).toEqual([1])
    expect(() => update_parameters(objective, gradient(objective, "value"), sgd())).toThrow("source parameters")

    result.dispose()
    value.dispose()
    state.dispose()
    session.dispose()
  })

  test("combines reusable model and loss Programs while preserving inference state", () => {
    const model = program("reusable_classifier", p => {
      const x = p.argument("x", Tensor.f32([2, 2]))
      return linear({ out_features: 2 })({ x }, "head")
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

  test("materializes specialized loss callables from affon:nn", () => {
    const head = linear({ out_features: 2 })
    const model = program("builtin_loss_model", p => head({ x: p.argument("x", Tensor.f32([2, 2])) }, "head"))
    const optimizer = sgd({ learning_rate: 0.01 })
    const lossCallables = [
      crossEntropyLoss(),
      mean_squared_error(),
      binary_cross_entropy(),
      binary_cross_entropy_with_logits(),
    ]
    expect(lossCallables.every(Object.isFrozen)).toBe(true)
    const steps = lossCallables.map(loss => optimize(model, loss, optimizer))
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
    const renamed = optimize(model, crossEntropyLoss({ target: "class_id" }), optimizer)
    expect(renamed.inspect().arguments.at(-1)?.name).toBe("class_id")
    expect(() => crossEntropyLoss({ reduction: "sum" as any })).toThrow("mean reduction")

    const customLoss: LossCallable = ({ input, target }) => {
      const difference = sub(input, target)
      return mean(mul(difference, difference))
    }
    const customStep = optimize(model, customLoss, optimizer)
    expect(customStep.inspect().arguments.map(value => value.name)).toEqual(["x", "target"])
  })

  test("composes a loss callable directly with named bindings", () => {
    const objective = crossEntropyLoss()
    const source = program("callable_loss", p => objective({
      input: p.argument("input", Tensor.f32([2, 3])),
      target: p.argument("target", Tensor.i64([2])),
    }, "objective"))

    expect(source.inspect().nodes.at(-1)?.op).toBe("cross_entropy")
    expect(source.inspect().outputs.length).toBe(1)
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

})
