import { getEnv } from "std:process"
import { describe, expect, test } from "std:test"
import { add, embedding, gather, index_select, matmul, neg, permute, reduce_to_shape, reshape, scatter_add, sin, slice, squeeze, sum, transpose, unsqueeze } from "affon:ops"
import { Session, Tensor, program } from "affon:core"

const backendSymbol = Symbol.for("affon.compute.executable_backend")
const routingSymbol = Symbol.for("affon.compute.executable_routing")
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
      expect((denseGuarded as any)[backendSymbol]()).toBe("cyna")
      const routing = (executable as any)[routingSymbol]()
      expect(routing.schema_version).toBe(1)
      expect(routing.backend).toBe("cyna")
      expect(routing.configured_backend).toBe("cpu")
      expect(routing.category).toBe("success")
      expect(routing.reason).toBe("lowered")
      expect(routing.operation).toBe(null)
      expect(routing.option).toBe(null)
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

  test("enforces dense-input guards on the Cyna artifact", () => {
    const source = program("cyna_dense_guard", p => sin(p.argument("x", Tensor.f32([2, 2]))))
    const session = new Session({ device: "cpu" })
    const dense = session.compile(source, { denseInputs: true })
    const generic = session.compile(source)
    if (requireCyna) {
      expect((generic as any)[backendSymbol]()).toBe("cyna")
      expect((dense as any)[backendSymbol]()).toBe("cyna")
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

  test("routes identity Programs through Cyna", () => {
    const identity = program("cyna_identity", p => p.argument("value", Tensor.f32([2])))
    const session = new Session({ device: "cpu" })
    const identityExecutable = session.compile(identity)
    if (requireCyna) expect((identityExecutable as any)[backendSymbol]()).toBe("cyna")

    const value = session.tensor([2, -3])
    const identityResult = identityExecutable.run({ value }) as any
    expect(identityResult.to_array()).toEqual([2, -3])

    identityResult.dispose()
    value.dispose()
    identityExecutable.dispose()
    session.dispose()
  })

  test("routes repeated binary operands through Cyna", () => {
    const source = program("cyna_repeated_operand", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return add(value, value)
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    if (requireCyna) {
      expect((executable as any)[routingSymbol]().reason).toBe("lowered")
      expect((executable as any)[backendSymbol]()).toBe("cyna")
    }

    const value = session.tensor([2, -3])
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([4, -6])

    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("routes negation through Cyna", () => {
    const source = program("cyna_neg", p => neg(p.argument("value", Tensor.f32([2]))))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    if (requireCyna) expect((executable as any)[backendSymbol]()).toBe("cyna")

    const value = session.tensor([2, -3])
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([-2, 3])

    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("reports unsupported lowering details before using the Zig fallback", () => {
    const source = program("cyna_unsupported_reduce_shape", p => reduce_to_shape(
      p.argument("value", Tensor.f32([2, 3])),
      [1, 3],
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    const routing = (executable as any)[routingSymbol]()
    expect(routing.configured_backend).toBe("cpu")
    if (requireCyna) {
      expect(routing.backend).toBe("zig")
      expect(routing.category).toBe("operation")
      expect(routing.reason).toBe("InvalidProgram")
      expect(routing.operation).toBe("reduce_to_shape")
      expect(routing.option).toBe("reduce_to_shape")
    } else expect(routing.reason).toBe("cyna_disabled")
    executable.dispose()
    session.dispose()
  })

  test("routes axis and whole keepdim reductions through Cyna", () => {
    const source = program("cyna_keepdim_reductions", p => {
      const value = p.argument("value", Tensor.f32([2, 2]))
      return [sum(value, 1, true), sum(value, undefined, true)]
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    if (requireCyna) {
      expect((executable as any)[routingSymbol]().reason).toBe("lowered")
      expect((executable as any)[backendSymbol]()).toBe("cyna")
    }

    const value = session.tensor([[1, 2], [3, 4]])
    const outputs = executable.run({ value }) as any[]
    expect(outputs[0].shape).toEqual([2, 1])
    expect(outputs[0].to_array()).toEqual([[3], [7]])
    expect(outputs[1].shape).toEqual([1, 1])
    expect(outputs[1].to_array()).toEqual([[10]])

    for (const output of outputs) output.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("reports evidence capture as a routing fallback", () => {
    const source = program("cyna_evidence_fallback", p => sin(
      p.argument("value", Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 32 } })
    const executable = session.compile(source)
    const routing = (executable as any)[routingSymbol]()
    if (requireCyna) {
      expect(routing.backend).toBe("zig")
      expect(routing.category).toBe("observability")
      expect(routing.reason).toBe("evidence_capture")
    }
    executable.dispose()
    session.dispose()
  })

  test("reports compute timing as a routing fallback", () => {
    const source = program("cyna_timing_fallback", p => sin(
      p.argument("value", Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu", telemetry: { computeTiming: "invocation" } })
    const executable = session.compile(source)
    const routing = (executable as any)[routingSymbol]()
    if (requireCyna) {
      expect(routing.backend).toBe("zig")
      expect(routing.category).toBe("observability")
      expect(routing.reason).toBe("compute_evidence")
    }
    executable.dispose()
    session.dispose()
  })

  test("returns public layout views from Cyna without losing stride guards", () => {
    const source = program("cyna_public_layout_views", p => {
      const matrix = p.argument("matrix", Tensor.f32([2, 3]))
      const batch = p.argument("batch", Tensor.f32([2, 3, 1]))
      const padded = p.argument("padded", Tensor.f32([1, 2, 1]))
      const wide = p.argument("wide", Tensor.f32([3, 4]))
      return [
        transpose(matrix, [1, 0]),
        permute(batch, [2, 0, 1]),
        reshape(matrix, [3, 2]),
        squeeze(padded),
        unsqueeze(matrix, 1),
        slice(wide, [{ start: 1, stop: 3 }, { start: 1, stop: 4, step: 2 }]),
      ]
    })
    const denseSource = program("cyna_layout_guard", p => sin(
      p.argument("x", Tensor.f32([3, 2])),
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    const dense = session.compile(denseSource, { denseInputs: true })
    if (requireCyna) expect((executable as any)[backendSymbol]()).toBe("cyna")

    const matrix = session.tensor([[1, 2, 3], [4, 5, 6]])
    const batch = session.tensor([[[1], [2], [3]], [[4], [5], [6]]])
    const padded = session.tensor([[[7], [8]]])
    const wide = session.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    const wrong = session.tensor([1, 2, 3, 4, 5, 6])
    expect(() => executable.run({
      matrix: wrong as any,
      batch,
      padded,
      wide,
    })).toThrow("TensorSpec")
    const outputs = executable.run({ matrix, batch, padded, wide }) as any[]
    expect(outputs[0].to_array()).toEqual([[1, 4], [2, 5], [3, 6]])
    expect(outputs[1].to_array()).toEqual([[[1, 2, 3], [4, 5, 6]]])
    expect(outputs[2].to_array()).toEqual([[1, 2], [3, 4], [5, 6]])
    expect(outputs[3].to_array()).toEqual([7, 8])
    expect(outputs[4].to_array()).toEqual([[[1, 2, 3]], [[4, 5, 6]]])
    expect(outputs[5].to_array()).toEqual([[5, 7], [9, 11]])
    expect(() => dense.run({ x: outputs[0] })).toThrow("GuardMismatch")
    const denseInput = session.tensor([[1, 4], [2, 5], [3, 6]])
    dense.run({ x: denseInput }).dispose()

    denseInput.dispose()
    for (const output of outputs) output.dispose()
    wrong.dispose()
    wide.dispose()
    padded.dispose()
    batch.dispose()
    matrix.dispose()
    dense.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes indexed Programs through the persistent Cyna graph", () => {
    const source = program("cyna_public_indexed_ops", p => {
      const input = p.argument("input", Tensor.f32([2, 3]))
      const rows = p.argument("rows", Tensor.i64([2]))
      const table = p.argument("table", Tensor.f32([4, 2]))
      const rowGrid = p.argument("row_grid", Tensor.i64([2, 3]))
      const gatherIndices = p.argument("gather_indices", Tensor.i64([2, 2]))
      const base = p.argument("base", Tensor.f32([2, 3]))
      const scatterIndices = p.argument("scatter_indices", Tensor.i64([2, 3]))
      const updates = p.argument("updates", Tensor.f32([2, 3]))
      return [
        index_select(input, 1, rows),
        embedding(table, rowGrid),
        gather(input, 1, gatherIndices),
        scatter_add(base, scatterIndices, updates, 0),
      ]
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    if (requireCyna) expect((executable as any)[backendSymbol]()).toBe("cyna")

    const input = session.tensor([[10, 20, 30], [40, 50, 60]])
    const rows = session.tensor([2, 0], { dtype: "i64" })
    const table = session.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    const rowGrid = session.tensor([[3, 0, 3], [1, 2, 0]], { dtype: "i64" })
    const gatherIndices = session.tensor([[2, 0], [1, 1]], { dtype: "i64" })
    const base = session.tensor([[1, 2, 3], [4, 5, 6]])
    const scatterIndices = session.tensor([[0, 1, 1], [1, 0, 1]], { dtype: "i64" })
    const updates = session.tensor([[1, 1, 1], [2, 2, 2]])
    const outputs = executable.run({
      input,
      rows,
      table,
      row_grid: rowGrid,
      gather_indices: gatherIndices,
      base,
      scatter_indices: scatterIndices,
      updates,
    }) as any[]
    expect(outputs[0].to_array()).toEqual([[30, 10], [60, 40]])
    expect(outputs[1].to_array()).toEqual([[[7, 8], [1, 2], [7, 8]], [[3, 4], [5, 6], [1, 2]]])
    expect(outputs[2].to_array()).toEqual([[30, 10], [50, 50]])
    expect(outputs[3].to_array()).toEqual([[2, 4, 3], [6, 6, 9]])

    for (const output of outputs) output.dispose()
    updates.dispose()
    scatterIndices.dispose()
    base.dispose()
    gatherIndices.dispose()
    rowGrid.dispose()
    table.dispose()
    rows.dispose()
    input.dispose()
    executable.dispose()
    session.dispose()
  })

  test("preserves indexed failures and reusable executables on Cyna", () => {
    const selectedSource = program("cyna_public_index_select_bounds", p => index_select(
      p.argument("input", Tensor.f32([2, 2])),
      0,
      p.argument("indices", Tensor.i64([1])),
    ))
    const scatteredSource = program("cyna_public_scatter_bounds", p => scatter_add(
      p.argument("base", Tensor.f32([2, 2])),
      p.argument("indices", Tensor.i64([1, 2])),
      p.argument("updates", Tensor.f32([1, 2])),
      0,
    ))
    const session = new Session({ device: "cpu" })
    const selected = session.compile(selectedSource)
    const scattered = session.compile(scatteredSource)
    if (requireCyna) {
      expect((selected as any)[backendSymbol]()).toBe("cyna")
      expect((scattered as any)[backendSymbol]()).toBe("cyna")
    }

    const input = session.tensor([[1, 2], [3, 4]])
    const invalidRow = session.tensor([2], { dtype: "i64" })
    const validRow = session.tensor([1], { dtype: "i64" })
    expect(() => selected.run({ input, indices: invalidRow })).toThrow("InvalidCommand")
    const recoveredSelection = selected.run({ input, indices: validRow }) as any
    expect(recoveredSelection.to_array()).toEqual([[3, 4]])

    const base = session.tensor([[1, 2], [3, 4]])
    const invalidScatter = session.tensor([[2, 0]], { dtype: "i64" })
    const validScatter = session.tensor([[1, 0]], { dtype: "i64" })
    const updates = session.tensor([[5, 6]])
    expect(() => scattered.run({ base, indices: invalidScatter, updates })).toThrow("InvalidCommand")
    const recoveredScatter = scattered.run({ base, indices: validScatter, updates }) as any
    expect(recoveredScatter.to_array()).toEqual([[1, 8], [8, 4]])

    recoveredScatter.dispose()
    recoveredSelection.dispose()
    updates.dispose()
    validScatter.dispose()
    invalidScatter.dispose()
    base.dispose()
    validRow.dispose()
    invalidRow.dispose()
    input.dispose()
    selected.dispose()
    scattered.dispose()
    session.dispose()
  })
})
