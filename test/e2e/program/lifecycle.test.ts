import { add } from "affon:ops"
import { describe, expect, test } from "std:test"
import { Session, Tensor, program } from "affon:compute"

describe("Program execution lifecycle", () => {
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

})
