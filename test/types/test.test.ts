import testModule, {
  captureOutput,
  describe,
  expect,
  mock,
  test,
  values,
} from "std:test"

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
function assertType<T extends true>() {}

describe("std:test types", () => {
  test("exports public test helpers", () => {
    expect(values([1, 2])).toEqual([1, 2])
  })
})

const fn = mock.fn((value: number) => value + 1)
const captured = captureOutput(() => {
  expect(fn(1)).toBe(2)
})

assertType<IsExact<typeof testModule.describe, typeof describe>>()
assertType<IsExact<typeof testModule.test, typeof test>>()
assertType<IsExact<typeof fn.callCount, number>>()
assertType<IsExact<typeof captured, Promise<{ stdout: string; stderr: string; combined: string; exitCode: number | null; signal: number | null }>>>()
