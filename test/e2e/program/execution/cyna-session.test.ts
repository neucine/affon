import { getEnv } from "std:process"
import { describe, expect, test } from "std:test"
import { add, matmul, sin, transpose } from "affon:ops"
import { Session, Tensor, program } from "affon:core"

const backendSymbol = Symbol.for("affon.compute.executable_backend")
const requireCyna = getEnv("AFFON_TEST_CYNA_SESSION") === "1"

describe("public CPU Session Cyna execution", () => {
  test("executes a covered Program through the persistent Cyna graph", () => {
    const source = program("cyna_public_matmul_add", p => add(
      add(
        matmul(
          p.argument("left", Tensor.f32([2, 3])),
          p.argument("right", Tensor.f32([3, 2])),
        ),
        p.argument("bias", Tensor.f32([2, 2])),
      ),
      p.constant("offset", [[1, 1], [2, 2]], Tensor.f32([2, 2])),
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    const denseGuarded = session.compile(source, { denseInputs: true })
    const backend = (executable as any)[backendSymbol]()
    expect(["cyna", "zig"]).toContain(backend)
    if (requireCyna) {
      expect(backend).toBe("cyna")
      expect((denseGuarded as any)[backendSymbol]()).toBe("zig")
    }

    const leftSource = session.tensor([[1, 4], [2, 5], [3, 6]])
    const left = transpose(leftSource, [1, 0])
    const right = session.tensor([[1, 2], [3, 4], [5, 6]])
    const bias = session.tensor([[10, 20], [30, 40]])
    const wrong = session.tensor([1, 2, 3])
    expect(() => executable.run({ left: wrong as any, right, bias })).toThrow("TensorSpec")
    const output = executable.run({ left, right, bias }) as any
    expect(output.shape).toEqual([2, 2])
    expect(output.dtype).toBe("f32")
    expect(output.to_bytes().byteLength).toBe(16)
    expect(output.to_array()).toEqual([[33, 49], [81, 106]])

    output.dispose()
    bias.dispose()
    right.dispose()
    left.dispose()
    leftSource.dispose()
    wrong.dispose()
    session.dispose()
  })

  test("keeps dense-input guards on the Zig artifact", () => {
    const source = program("cyna_dense_guard", p => sin(p.argument("x", Tensor.f32([2, 2]))))
    const session = new Session({ device: "cpu" })
    const dense = session.compile(source, { denseInputs: true })
    const generic = session.compile(source)
    if (requireCyna) {
      expect((generic as any)[backendSymbol]()).toBe("cyna")
      expect((dense as any)[backendSymbol]()).toBe("zig")
    }
    const input = session.tensor([[0, 1], [2, 3]])
    const view = transpose(input, [1, 0])
    dense.run({ x: input }).dispose()
    expect(() => dense.run({ x: view })).toThrow("GuardMismatch")
    generic.run({ x: view }).dispose()
    view.dispose()
    input.dispose()
    session.dispose()
  })
})
