import { add, cat, contiguous, embedding, index_select, layer_norm, masked_fill, mean, mul, slice, squeeze, unsqueeze } from "affon:ops"
import { describe, expect, test } from "std:test"
import { Session, Tensor, gradient, losses, optimize, program } from "affon:compute"
import { adam } from "affon:optim"

describe("Program execution state and lowering", () => {
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
