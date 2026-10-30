import { describe, expect, test } from "std:test"
import { Session, Tensor, program } from "affon:compute"
import * as compute from "affon:compute"
import * as nn from "affon:nn"
import * as optim from "affon:optim"

describe("Program public surface and values", () => {
  test("exports only constructible public runtime entry points", () => {
    expect((compute as any).default).toBe(undefined)
    expect((nn as any).default).toBe(undefined)
    expect((optim as any).default).toBe(undefined)
    expect(typeof nn.linear).toBe("function")
    expect(typeof nn.embedding).toBe("function")
    expect(typeof nn.layer_norm).toBe("function")
    expect(typeof nn.cross_entropy).toBe("function")
    expect(typeof nn.mean_squared_error).toBe("function")
    expect(typeof nn.binary_cross_entropy).toBe("function")
    expect(typeof nn.binary_cross_entropy_with_logits).toBe("function")
    expect((compute as any).losses).toBe(undefined)
    expect(typeof compute.gradient).toBe("function")
    expect(typeof compute.update_parameters).toBe("function")
    expect(typeof compute.optimize).toBe("function")
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

  test("bounds JavaScript array tensors and accepts direct numeric bytes", () => {
    const session = new Session({ device: "cpu" })
    expect(() => session.tensor(new Array(262145).fill(0))).toThrow("Tensor.fromBytes")
    expect(() => program("oversized_constant", p => p.constant("values", new Array(262145).fill(0), Tensor.f32([262145])))).toThrow("ProgramBuilder.full")
    const bytes = new Float32Array([1.25, -2.5, 3.75, 0]).buffer
    const value = session.fromBytes(bytes, Tensor.f32([2, 2]))
    expect(value.to_array()).toEqual([[1.25, -2.5], [3.75, 0]])
    expect(() => session.fromBytes(bytes, Tensor.f32([5]))).toThrow("byte length")
    value.dispose()
    session.dispose()
  })

  test("compiles uniform shaped constants without nested arrays", () => {
    const source = program("filled_constant", p => p.full("values", 2.5, Tensor.f32([2, 3])))
    expect(source.inspect().nodes.find(node => node.role === "constant")?.value).toBe(2.5)
    const session = new Session({ device: "cpu" })
    const output = session.compile(source).run({}) as any
    expect(output.to_array()).toEqual([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]])
    output.dispose()
    session.dispose()
    const comparison = program("filled_vs_array", p => [
      p.full("filled", 0, Tensor.f32([64, 64])),
      p.constant("array", Array.from({ length: 64 }, () => Array(64).fill(0)), Tensor.f32([64, 64])),
    ])
    const paritySession = new Session({ device: "cpu" })
    const [filled, fromArray] = paritySession.compile(comparison).run({}) as any
    expect(filled.to_array()).toEqual(fromArray.to_array())
    filled.dispose()
    fromArray.dispose()
    paritySession.dispose()
    const empty = program("empty_fill", p => p.full("empty", 0, Tensor.f32([0])))
    const emptySession = new Session({ device: "cpu" })
    const emptyOutput = emptySession.compile(empty).run({}) as any
    expect(emptyOutput.to_array()).toEqual([])
    emptyOutput.dispose()
    emptySession.dispose()
  })

})
