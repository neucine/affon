import { getEnv } from "std:process"
import { describe, expect, test } from "std:test"
import { add, embedding, gather, index_select, matmul, scatter_add, sin, transpose } from "affon:ops"
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
