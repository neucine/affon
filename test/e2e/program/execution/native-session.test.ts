import telemetry from "std:telemetry"
import { describe, expect, test } from "std:test"
import { add, cross_entropy, dense_cross_entropy, div, dot, embedding, gather, gelu, index_select, log_softmax_nll, matmul, mul, neg, permute, quantized_matmul, reduce_to_shape, reshape, scatter_add, sin, slice, squeeze, sub, sum, topk, transpose, unsqueeze } from "affon:ops"
import { Session, Tensor, program } from "affon:core"


describe("public CPU Session execution", () => {
  test("rejects the removed telemetry alias", () => {
    expect(() => new Session({ telemetry: { backendTiming: true } as any }))
      .toThrow("backendTiming has been removed; use computeTiming")
  })

  test("attributes one prepared top-k ranking to both outputs", () => {
    const source = program("paired_topk", p => topk(p.argument("x", Tensor.f32([2, 4])), 2, 1))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source)
    const x = session.tensor([[3, 5, 5, 1], [2, 2, 4, 3]])
    const result = executable.run({ x }) as any[]
    expect(result[0].to_array()).toEqual([[5, 5], [4, 3]])
    expect(result[1].to_array()).toEqual([[1, 2], [2, 3]])
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.entries.map((entry: any) => entry.operation))
        .toEqual(["top_k_values", "top_k_values"])
      expect(evidence.entries.map((entry: any) => entry.origins))
        .toEqual([[0], [1]])
      expect((executable.explain() as any).selected_physical_nodes
        .map((node: any) => node.implementation))
        .toEqual(["cpu_indexing_family", "cpu_indexing_family"])
      expect(evidence.entries.map((entry: any) =>
        entry.physical_output_bytes)).toEqual([16, 32])
      expect(evidence.entries[0].invocation_timing_sample_count).toBe(1)
      expect(evidence.entries[0].kernel_count).toBe(1)
      expect(evidence.kernel_count_evidence).toBe("exact")
    }
    result.forEach(tensor => tensor.dispose())
    x.dispose(); executable.dispose(); session.dispose()
  })

  test("reports unavailable CPU hardware and profiler telemetry through Compute Core", () => {
    const source = program("profile_observation", p => sin(p.argument("x", Tensor.f32([2]))))
    for (const item of [
      {
        option: { hardwareMetrics: true },
        check: (evidence: any) => {
          expect(evidence.profiler_marker_availability).toBe(null)
          expect(evidence.hardware_metric_availability).toEqual([
            { metric: "bandwidth", availability: "unsupported" },
            { metric: "occupancy", availability: "unsupported" },
            { metric: "cache", availability: "unsupported" },
          ])
        },
      },
      {
        option: { profilerMarkers: true },
        check: (evidence: any) => {
          expect(evidence.hardware_metric_availability).toEqual([
            { metric: "bandwidth", availability: "unsupported" },
            { metric: "occupancy", availability: "unsupported" },
            { metric: "cache", availability: "unsupported" },
          ])
          expect(evidence.profiler_marker_availability).toEqual({
            provider: "nvtx",
            available: false,
          })
        },
      },
    ]) {
      const session = new Session({
        device: "cpu",
        telemetry: { ...item.option, captureRecords: 64 },
      })
      const executable = session.compile(source)
      const value = session.tensor([0, 1])
      const result = executable.run({ x: value }) as any
      expect(result.to_array()).toHaveLength(2)
      item.check(session.runtimeEvidence(executable))
      result.dispose()
      value.dispose()
      executable.dispose()
      session.dispose()
    }
  })

  test("executes a covered Program through the in-process Compute Core Program", () => {
    const source = program("public_matmul_add", p => add(
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

    const repeated = executable.run({ left, right, bias }) as any
    expect(repeated.to_array()).toEqual([[33, 49], [81, 106]])
    const changedLeft = session.tensor([[2, 3, 4], [5, 6, 7]])
    const changed = executable.run({ left: changedLeft, right, bias }) as any
    expect(changed.to_array()).toEqual([[42, 61], [90, 118]])

    changed.dispose()
    changedLeft.dispose()
    repeated.dispose()
    output.dispose()
    bias.dispose()
    right.dispose()
    left.dispose()
    leftSource.dispose()
    wrong.dispose()
    session.dispose()
  })

  test("enforces dense-input guards on the Compute Core artifact", () => {
    const source = program("dense_guard", p => sin(p.argument("x", Tensor.f32([2, 2]))))
    const session = new Session({ device: "cpu" })
    const dense = session.compile(source, { denseInputs: true })
    const generic = session.compile(source)
    const input = session.tensor([[0, 1], [2, 3]])
    const view = transpose(input, [1, 0])
    dense.run({ x: input }).dispose()
    expect(() => dense.run({ x: view })).toThrow("GuardMismatch")
    generic.run({ x: view }).dispose()
    view.dispose()
    input.dispose()
    session.dispose()
  })

  test("executes identity Programs through Compute Core", () => {
    const identity = program("identity", p => p.argument("value", Tensor.f32([2])))
    const session = new Session({ device: "cpu" })
    const identityExecutable = session.compile(identity)

    const value = session.tensor([2, -3])
    const identityResult = identityExecutable.run({ value }) as any
    expect(identityResult.to_array()).toEqual([2, -3])

    identityResult.dispose()
    value.dispose()
    identityExecutable.dispose()
    session.dispose()
  })

  test("executes repeated binary operands through Compute Core", () => {
    const source = program("repeated_operand", p => {
      const value = p.argument("value", Tensor.f32([2]))
      return add(value, value)
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)

    const value = session.tensor([2, -3])
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([4, -6])

    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes negation through Compute Core", () => {
    const source = program("neg", p => neg(p.argument("value", Tensor.f32([2]))))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)

    const value = session.tensor([2, -3])
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([-2, 3])

    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes reduce_to_shape through Compute Core", () => {
    const source = program("reduce_shape", p => reduce_to_shape(
      p.argument("value", Tensor.f32([2, 3])),
      [1, 3],
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)

    const value = session.tensor([[1, 2, 3], [4, 5, 6]])
    const result = executable.run({ value }) as any
    expect(result.shape).toEqual([1, 3])
    expect(result.to_array()).toEqual([[5, 7, 9]])

    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes quantized matmul through Compute Core", () => {
    const source = program("quantized_matmul", p => quantized_matmul(
      p.argument("x", Tensor.f32([1, 4])),
      p.argument("packed", Tensor.u32([1, 1])),
      p.argument("scales", Tensor.bf16([1, 1])),
      p.argument("biases", Tensor.bf16([1, 1])),
      { bits: 8, group_size: 4 },
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)

    const x = session.tensor([[1, 2, 3, 4]])
    const packed = session.tensor([[0x04030201]], { dtype: "u32" })
    const scales = session.tensor([[1]], { dtype: "bf16" })
    const biases = session.tensor([[0]], { dtype: "bf16" })
    const result = executable.run({ x, packed, scales, biases }) as any
    expect(result.to_array()).toEqual([[30]])

    result.dispose()
    biases.dispose()
    scales.dispose()
    packed.dispose()
    x.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes indexed and dense losses through Compute Core", () => {
    const source = program("losses", p => {
      const logits = p.argument("logits", Tensor.f32([2, 3]))
      const targets = p.argument("targets", Tensor.f32([2, 3]))
      return [
        cross_entropy(logits, p.argument("labels", Tensor.i64([2]))),
        dense_cross_entropy(logits, targets),
        log_softmax_nll(logits, targets),
      ]
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)

    const logits = session.tensor([[0, 0, 0], [0, 0, 0]])
    const labels = session.tensor([0, 2], { dtype: "i64" })
    const targets = session.tensor([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    const results = executable.run({ logits, labels, targets }) as any[]
    for (const result of results) {
      expect(result.shape).toEqual([1])
      expect(Math.abs((result.to_array() as number[])[0] - Math.log(3)) < 1e-6).toBe(true)
    }

    for (const result of results) result.dispose()
    targets.dispose()
    labels.dispose()
    logits.dispose()
    executable.dispose()
    session.dispose()
  })

  test("executes axis and whole keepdim reductions through Compute Core", () => {
    const source = program("keepdim_reductions", p => {
      const value = p.argument("value", Tensor.f32([2, 2]))
      return [sum(value, 1, true), sum(value, undefined, true)]
    })
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    {
      expect((executable.explain() as any).selected_physical_nodes.map((node: any) => node.operation).sort()).toEqual([
        "sum_all", "sum_axis",
      ])
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

  test("runs evidence capture through Compute Core", () => {
    const source = program("evidence_capture", p => sin(
      p.argument("value", Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 32 } })
    const executable = session.compile(source)
    const value = session.tensor([0, 1])
    const result = executable.run({ value }) as any
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.backend).toBe("cpu")
      expect(evidence.entries[0].operation).toBe("sin")
      expect(evidence.entries[0].invocation_count).toBe(1)
    }
    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("runs invocation and kernel timing through Compute Core", () => {
    const source = program("timing_observation", p => sin(
      p.argument("value", Tensor.f32([2])),
    ))
    for (const computeTiming of ["invocation", "kernel"] as const) {
      const session = new Session({ device: "cpu", telemetry: { computeTiming, captureRecords: 64 } })
      const executable = session.compile(source)
      const value = session.tensor([0, 1])
      const runsBefore = telemetry.metrics().find(metric => metric.scope === "compute.execution" && metric.name === "runs_total")?.value ?? 0
      const timeBefore = telemetry.metrics().find(metric => metric.scope === "compute.backend" && metric.name === "elapsed_nanoseconds_total")?.value ?? 0
      const result = executable.run({ value }) as any
      expect(result.to_array()[0]).toBe(0)
      {
        expect(telemetry.metrics().find(metric => metric.scope === "compute.execution" && metric.name === "runs_total")?.value).toBe(runsBefore + 1)
        expect((telemetry.metrics().find(metric => metric.scope === "compute.backend" && metric.name === "elapsed_nanoseconds_total")?.value ?? 0) > timeBefore).toBe(true)
        const evidence = session.runtimeEvidence(executable) as any
        expect(evidence.timing_scope).toBe(computeTiming)
        expect(evidence.entries[0].timing_scope).toBe(computeTiming)
        if (computeTiming === "kernel") {
          expect(evidence.entries[0].kernel_timing_sample_count).toBe(1)
          expect(evidence.entries[0].invocation_timing_sample_count).toBe(0)
        } else {
          expect(evidence.entries[0].invocation_timing_sample_count).toBe(1)
          expect(evidence.entries[0].kernel_timing_sample_count).toBe(0)
        }
      }
      result.dispose()
      value.dispose()
      executable.dispose()
      session.dispose()
    }
  })

  test("keeps Compute Core evidence across runs and clears it on reset", () => {
    const source = program("evidence_repeated", p => add(
      p.argument("left", Tensor.f32([2])), p.argument("right", Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source)
    const left = session.tensor([1, 2])
    const right = session.tensor([3, 4])
    session.resetEvidence()
    for (let index = 0; index < 2; index++) (executable.run({ left, right }) as any).dispose()
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.backend).toBe("cpu")
      expect(evidence.observed_run_count).toBe(2)
      expect(evidence.dropped_event_count).toBe(0)
      expect(evidence.entries[0].operation).toBe("add")
      expect(evidence.entries[0].invocation_count).toBe(2)
      expect(evidence.entries[0].observed_schedule_position).toBe(0)
      expect(evidence.entries[0].invocation_timing_sample_count).toBe(2)
      expect(evidence.entries[0].kernel_count).toBe(2)
      expect(evidence.kernel_count_evidence).toBe("exact")
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.prepared_cache_misses).toBe(1)
      expect(evidence.prepared_cache_hits >= 1).toBe(true)
      expect(evidence.entries[0].physical_action).toBe("instruction")
      expect(typeof evidence.entries[0].physical_is_view).toBe("boolean")
    }
    session.resetEvidence()
    const cleared = session.runtimeEvidence(executable) as any
    {
      expect(cleared.observed_run_count).toBe(0)
      expect(cleared.entries[0].invocation_count).toBe(0)
      expect(cleared.entries[0].observed_schedule_position).toBe(null)
      expect(cleared.entries[0].kernel_count).toBe(null)
      expect(cleared.dropped_event_count).toBe(0)
    }
    left.dispose()
    right.dispose()
    executable.dispose()
    session.dispose()
  })

  test("reports the selected memory graph and prepared reuse", () => {
    const source = program("memory_plan", p => add(
      matmul(p.argument("left", Tensor.f32([2, 3])),
        transpose(p.argument("right", Tensor.f32([4, 3])), [1, 0])),
      p.argument("bias", Tensor.f32([2, 4])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source, { optimizationGoal: "memory" })
    const selectedNodes = (executable.explain() as any).selected_physical_nodes
    {
      const explanation = executable.explain() as any
      expect(explanation.execution_backend).toBe("compute_core")
      expect(explanation.executable_plan_backend).toBe("compute_core")
      expect(explanation.selected_semantic_steps.some((step: any) => step.implementation_code === 1)).toBe(true)
      expect(explanation.residual_decisions).toEqual([])
      expect(selectedNodes.map((node: any) => node.implementation)).toEqual([
        "cpu_structural", "cpu_fused_matmul_add",
      ])
    }
    const left = session.tensor([[1, 2, 3], [4, 5, 6]])
    const right = session.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    const bias = session.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    const expected = [[15, 33, 51, 69], [33, 78, 123, 168]]
    for (let index = 0; index < 2; index++) {
      const result = executable.run({ left, right, bias }) as any
      expect(result.to_array()).toEqual(expected)
      result.dispose()
    }
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.prepared_cache_misses).toBe(1)
      expect(evidence.prepared_cache_hits >= 1).toBe(true)
      expect(evidence.prepared_cache_bytes).toBe(32)
      expect(evidence.entries.map((entry: any) => entry.physical_implementation)).toEqual([
        "cpu_structural", "cpu_fused_matmul_add",
      ])
      expect(evidence.entries.map((entry: any) => entry.physical_implementation)).toEqual(
        selectedNodes.map((node: any) => node.implementation))
      expect(evidence.entries[0].physical_is_view).toBe(true)
      expect(evidence.entries[0].materialized_view_count).toBe(0)
      expect(evidence.entries[0].physical_output_bytes).toBe(48)
      expect(evidence.entries[0].materialized_view_bytes).toBe(0)
      expect(evidence.entries[0].kernel_count).toBe(0)
      expect(evidence.entries[1].physical_implementation)
        .toBe("cpu_fused_matmul_add")
    }
    left.dispose(); right.dispose(); bias.dispose(); executable.dispose(); session.dispose()
  })

  test("uses a transposed right matrix directly for batched matmul", () => {
    const source = program("batched_transposed_rhs", p => matmul(
      p.argument("left", Tensor.f32([1, 2, 3])),
      transpose(p.argument("right", Tensor.f32([4, 3])), [1, 0]),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 16 } })
    const executable = session.compile(source)
    const left = session.tensor([[[1, 2, 3], [4, 5, 6]]])
    const right = session.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    for (let index = 0; index < 2; index++) {
      const result = executable.run({ left, right }) as any
      expect(result.to_array()).toEqual([[[14, 32, 50, 68], [32, 77, 122, 167]]])
      result.dispose()
    }
    const changedRight = session.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]])
    const changed = executable.run({ left, right: changedRight }) as any
    expect(changed.to_array()).toEqual([[[20, 38, 56, 74], [47, 92, 137, 182]]])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.prepared_cache_misses).toBe(1)
      expect(evidence.prepared_cache_hits >= 2).toBe(true)
      expect(evidence.prepared_cache_bytes).toBe(32)
      expect(evidence.entries[0].physical_is_view).toBe(true)
      expect(evidence.entries[0].materialized_view_count).toBe(0)
      expect(evidence.entries[0].materialized_view_bytes).toBe(0)
    }
    changed.dispose(); changedRight.dispose(); left.dispose(); right.dispose(); executable.dispose(); session.dispose()
  })

  test("uses a transposed f64 right matrix directly for batched matmul", () => {
    const source = program("batched_transposed_f64_rhs", p => matmul(
      p.argument("left", Tensor.f64([1, 2, 3])),
      transpose(p.argument("right", Tensor.f64([4, 3])), [1, 0]),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 16 } })
    const executable = session.compile(source)
    const left = session.tensor([[[1, 2, 3], [4, 5, 6]]], { dtype: "f64" })
    const right = session.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], { dtype: "f64" })
    const result = executable.run({ left, right }) as any
    expect(result.to_array()).toEqual([[[14, 32, 50, 68], [32, 77, 122, 167]]])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.entries[0].materialized_view_count).toBe(0)
      expect(evidence.entries[0].kernel_count).toBe(0)
    }
    result.dispose(); left.dispose(); right.dispose(); executable.dispose(); session.dispose()
  })

  test("uses transposed right batches directly with matching and broadcast dimensions", () => {
    const cases = [
      {
        name: "matching", leftShape: [2, 1, 3],
        left: [[[1, 2, 3]], [[4, 5, 6]]],
        rightShape: [2, 2, 3],
        right: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        expected: [[[14, 32]], [[122, 167]]],
      },
      {
        name: "left_broadcast", leftShape: [1, 1, 3],
        left: [[[1, 2, 3]]],
        rightShape: [2, 2, 3],
        right: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        expected: [[[14, 32]], [[50, 68]]],
      },
      {
        name: "unbatched_left", leftShape: [1, 3],
        left: [[1, 2, 3]],
        rightShape: [2, 2, 3],
        right: [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        expected: [[[14, 32]], [[50, 68]]],
      },
      {
        name: "right_broadcast", leftShape: [2, 1, 3],
        left: [[[1, 2, 3]], [[4, 5, 6]]],
        rightShape: [1, 2, 3],
        right: [[[1, 2, 3], [4, 5, 6]]],
        expected: [[[14, 32]], [[32, 77]]],
      },
    ]
    for (const item of cases) {
      const source = program(`batched_transposed_${item.name}`, p => matmul(
        p.argument("left", Tensor.f32(item.leftShape)),
        transpose(p.argument("right", Tensor.f32(item.rightShape)), [0, 2, 1]),
      ))
      const session = new Session({ device: "cpu", telemetry: { captureRecords: 16 } })
      const executable = session.compile(source)
      const left = session.tensor(item.left)
      const right = session.tensor(item.right)
      const result = executable.run({ left, right }) as any
      expect(result.to_array()).toEqual(item.expected)
      {
        const evidence = session.runtimeEvidence(executable) as any
        expect(evidence.execution_path).toBe("prepared")
        expect(evidence.entries[0].materialized_view_count).toBe(0)
      }
      result.dispose(); left.dispose(); right.dispose(); executable.dispose(); session.dispose()
    }
  })

  test("uses transposed f64 right batches directly", () => {
    const source = program("batched_transposed_f64_right_batches", p => matmul(
      p.argument("left", Tensor.f64([1, 1, 3])),
      transpose(p.argument("right", Tensor.f64([2, 2, 3])), [0, 2, 1]),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 16 } })
    const executable = session.compile(source)
    const left = session.tensor([[[1, 2, 3]]], { dtype: "f64" })
    const right = session.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], { dtype: "f64" })
    const result = executable.run({ left, right }) as any
    expect(result.to_array()).toEqual([[[14, 32]], [[50, 68]]])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.entries[0].materialized_view_count).toBe(0)
    }
    result.dispose(); left.dispose(); right.dispose(); executable.dispose(); session.dispose()
  })

  test("aligns two broadcast batch axes for a direct right transpose", () => {
    const source = program("rank_four_transposed_rhs", p => matmul(
      p.argument("left", Tensor.f32([2, 1, 1, 3])),
      transpose(p.argument("right", Tensor.f32([1, 2, 2, 3])), [0, 1, 3, 2]),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 16 } })
    const executable = session.compile(source)
    const left = session.tensor([[[[1, 2, 3]]], [[[4, 5, 6]]]])
    const right = session.tensor([[
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]],
    ]])
    const result = executable.run({ left, right }) as any
    expect(result.to_array()).toEqual([
      [[[14, 32]], [[50, 68]]],
      [[[32, 77]], [[122, 167]]],
    ])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.entries[0].materialized_view_count).toBe(0)
    }
    result.dispose(); left.dispose(); right.dispose(); executable.dispose(); session.dispose()
  })

  test("passes each optimization goal to Compute Core plan selection", () => {
    const source = program("goal_selection", p => add(
      matmul(p.argument("left", Tensor.f32([2, 2])), p.argument("right", Tensor.f32([2, 2]))),
      p.argument("bias", Tensor.f32([2, 2])),
    ))
    for (const optimizationGoal of ["balanced", "latency", "throughput", "memory"] as const) {
      const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
      const executable = session.compile(source, { optimizationGoal })
      const left = session.tensor([[1, 2], [3, 4]])
      const right = session.tensor([[5, 6], [7, 8]])
      const bias = session.tensor([[1, 1], [1, 1]])
      const result = executable.run({ left, right, bias }) as any
      expect(result.to_array()).toEqual([[20, 23], [44, 51]])
      {
        const evidence = session.runtimeEvidence(executable) as any
        expect((executable.explain() as any).execution_backend).toBe("compute_core")
        const observedImplementations = evidence.entries.map((entry: any) => entry.physical_implementation)
        const selectedImplementations =
          (executable.explain() as any).selected_physical_nodes
            .map((node: any) => node.implementation)
        expect(observedImplementations).toEqual(selectedImplementations)
        expect(observedImplementations.length >= 1).toBe(true)
      }
      result.dispose(); left.dispose(); right.dispose(); bias.dispose(); executable.dispose(); session.dispose()
    }
  })

  test("fuses batched matmul plus bias under the memory goal", () => {
    for (const dtype of ["f32", "f64"] as const) {
      const spec = (shape: readonly number[]) => dtype === "f32" ? Tensor.f32(shape) : Tensor.f64(shape)
      const source = program(`batched_matmul_bias_${dtype}`, p => add(
        matmul(
          p.argument("left", spec([2, 3, 4])),
          p.argument("right", spec([2, 4, 5])),
        ),
        p.argument("bias", spec([5])),
      ))
      const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
      const executable = session.compile(source, { optimizationGoal: "memory" })
      const left = session.tensor([
        Array.from({ length: 3 }, () => [1, 1, 1, 1]),
        Array.from({ length: 3 }, () => [2, 2, 2, 2]),
      ], { dtype })
      const right = session.tensor([
        Array.from({ length: 4 }, () => [1, 1, 1, 1, 1]),
        Array.from({ length: 4 }, () => [1, 1, 1, 1, 1]),
      ], { dtype })
      const bias = session.tensor([1, 2, 3, 4, 5], { dtype })
      const output = executable.run({ left, right, bias }) as any
      expect(output.to_array()).toEqual([
        [[5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]],
        [[9, 10, 11, 12, 13], [9, 10, 11, 12, 13], [9, 10, 11, 12, 13]],
      ])
      {
        expect((executable.explain() as any).execution_backend).toBe("compute_core")
        const evidence = session.runtimeEvidence(executable) as any
        expect(evidence.execution_path).toBe("prepared")
        expect(evidence.entries.map((entry: any) => entry.physical_implementation))
          .toEqual(["cpu_fused_matmul_add"])
        expect((executable.explain() as any).selected_physical_nodes
          .filter((node: any) => node.source_instruction_count > 0)
          .map((node: any) => node.program_origins)).toEqual([[0, 1]])
      }
      output.dispose(); bias.dispose(); right.dispose(); left.dispose()
      executable.dispose(); session.dispose()
    }
  })

  test("selects add fusion families on Compute Core", () => {
    const cases = [
      program("add_add_goals", p => add(add(
        p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
        p.argument("c", Tensor.f32([2])))),
      program("add_gelu_goals", p => gelu(add(
        p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))))),
    ]
    for (const [caseIndex, source] of cases.entries()) for (const optimizationGoal of ["balanced", "latency", "throughput", "memory"] as const) {
      const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
      const executable = session.compile(source, { optimizationGoal })
      const a = session.tensor([1, 2]), b = session.tensor([3, 4]), c = session.tensor([5, 6])
      const result = executable.run(caseIndex === 0 ? { a, b, c } : { a, b }) as any
      {
        expect((executable.explain() as any).execution_backend).toBe("compute_core")
        const implementations = (session.runtimeEvidence(executable) as any).entries
          .map((entry: any) => entry.physical_implementation)
        if (caseIndex === 0 && optimizationGoal === "latency") {
          expect((executable.explain() as any).selected_physical_nodes
            .filter((node: any) => node.source_instruction_count > 0)
            .map((node: any) => node.origins)).toEqual([[0, 1]])
          expect((executable.explain() as any).selected_physical_nodes
            .filter((node: any) => node.source_instruction_count > 0)
            .map((node: any) => node.program_origins)).toEqual([[0, 1]])
        }
        expect(implementations.length >= 1).toBe(true)
        expect(implementations.every((id: string) => id.startsWith("cpu_"))).toBe(true)
      }
      result.dispose(); a.dispose(); b.dispose(); c.dispose(); executable.dispose(); session.dispose()
    }
  })

  test("applies public resource limits on the Compute Core compiler", () => {
    const source = program("resource_limits", p =>
      neg(p.argument("value", Tensor.f32([2]))))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
    expect(() => session.compile(source, {
      searchLimits: { maxPeakActiveBytes: 0 },
    })).toThrow()
    expect(() => session.compile(source, {
      searchLimits: { maxPeakActiveBytes: -1 },
    })).toThrow("invalid searchLimits.maxPeakActiveBytes")

    const executable = session.compile(source, {
      searchLimits: {
        maxPeakActiveBytes: 1024,
        maxInvocationScratchBytes: 0,
      },
    })
    {
      const search = (executable.explain() as any).search
      expect(search.limits.max_peak_active_bytes).toBe(1024)
      expect(search.limits.max_invocation_scratch_bytes).toBe(0)
    }
    const value = session.tensor([2, -3])
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([-2, 3])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.schema_version).toBe(11)
      expect(evidence.search.max_peak_active_bytes).toBe(1024)
      expect(evidence.search.max_invocation_scratch_bytes).toBe(0)
    }
    result.dispose(); value.dispose(); executable.dispose(); session.dispose()
  })

  test("selects and executes fused multiply then subtract", () => {
    for (const dtype of ["f32", "f64", "i64"] as const) {
      const spec = dtype === "f32" ? Tensor.f32([2]) : dtype === "f64" ? Tensor.f64([2]) : Tensor.i64([2])
      const source = program(`fused_mul_sub_${dtype}`, p => sub(
        p.argument("parameter", spec),
        mul(p.argument("gradient", spec), p.argument("rate", spec)),
      ))
      const parameterValues = dtype === "i64" ? [5, 6] : [100, 5]
      const gradientValues = dtype === "i64" ? [4294967296, 3] : [7, -8]
      const rateValues = dtype === "i64" ? [4294967296, 2] : [3, 4]
      const session = new Session({
        device: "cpu",
        telemetry: { captureRecords: 8 },
      })
      const executable = session.compile(source, { optimizationGoal: "balanced" })
      const parameter = session.tensor(parameterValues, { dtype })
      const gradient = session.tensor(gradientValues, { dtype })
      const rate = session.tensor(rateValues, { dtype })
      const output = executable.run({ parameter, gradient, rate }) as any
      expect(output.to_array()).toEqual(dtype === "i64" ? [5, 0] : [79, 37])
      {
        const explanation = executable.explain() as any
        expect(explanation.execution_backend).toBe("compute_core")
        expect(explanation.selected_physical_nodes.map(
          (node: any) => node.implementation
        )).toEqual(["cpu_fused_optimizer_update"])
        expect(explanation.selected_physical_nodes[0].program_origins)
          .toEqual([0, 1])
        expect((session.runtimeEvidence(executable) as any).execution_path)
          .toBe("prepared")
      }
      output.dispose(); parameter.dispose(); gradient.dispose(); rate.dispose()
      executable.dispose(); session.dispose()
    }
  })

  test("retains the unfused add chain when memory estimates tie", () => {
    const source = program("memory_add_choice", p => add(add(add(
      p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
      p.argument("c", Tensor.f32([2]))), p.argument("d", Tensor.f32([2]))))
    const session = new Session({ device: "cpu" })
    const selected = session.compile(source, { optimizationGoal: "memory" })
    const a = session.tensor([1, 2]), b = session.tensor([3, 4])
    const c = session.tensor([5, 6]), d = session.tensor([7, 8])
    const output = selected.run({ a, b, c, d }) as any
    expect(output.to_array()).toEqual([16, 20])
    {
      const actual = (selected.explain() as any).selected_physical_nodes
        .map((node: any) => node.implementation)
      const comparison = (selected.explain() as any).selection_comparison
      expect(comparison.evaluated_states).toBe(3)
      expect(comparison.budget).toBe(16)
      expect(comparison.exclusion_search_truncated).toBe(false)
      expect(comparison.candidate_peak_bytes).toBe(comparison.baseline_peak_bytes)
      expect(comparison.candidate_published).toBe(false)
      expect(actual).toEqual(["cpu_add", "cpu_add", "cpu_add"])
    }
    output.dispose(); a.dispose(); b.dispose(); c.dispose(); d.dispose()
    selected.dispose(); session.dispose()

    for (const length of [4, 5, 6]) {
      const longer = program(`memory_add_chain_${length}`, p => {
        let value = p.argument("x0", Tensor.f32([2]))
        for (let index = 1; index <= length; index++)
          value = add(value, p.argument(`x${index}`, Tensor.f32([2])))
        return value
      })
      const session = new Session({ device: "cpu" })
      const compiledPlan = session.compile(longer, { optimizationGoal: "memory" })
      {
        const actual = (compiledPlan.explain() as any).selected_physical_nodes
          .map((node: any) => node.implementation)
        expect(actual.every((implementation: string) =>
          implementation === "cpu_add")).toBe(true)
      }
      compiledPlan.dispose(); session.dispose()
    }
  })

  test("counts Program constant storage in a selected memory graph", () => {
    const source = program("memory_constant_accounting", p => add(
      p.argument("x", Tensor.f32([2])),
      p.constant("offset", [1, 2], Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source, { optimizationGoal: "memory" })
    const x = session.tensor([3, 4])
    const output = executable.run({ x }) as any
    expect(output.to_array()).toEqual([4, 6])
    {
      const comparison = (executable.explain() as any).selection_comparison
      expect(comparison.constant_input_bytes).toBe(8)
      const selected = executable.explain() as any
      expect(selected.selected_physical_values.map((value: any) => value.origin)).toContain("constant")
      expect(selected.selected_physical_liveness.peak_active_bytes).toBe(16)
    }
    output.dispose(); x.dispose(); executable.dispose(); session.dispose()
  })

  test("runs a constant-only Program through Compute Core", () => {
    const source = program("constant_only", p => p.constant(
      "values", [3, 5], Tensor.f32([2])))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    const output = executable.run({}) as any
    expect(output.to_array()).toEqual([3, 5])
    {
      expect((executable.explain() as any).selected_physical_values[0].origin).toBe("constant")
    }
    output.dispose(); executable.dispose(); session.dispose()
  })

  test("selects memory plans for multiply and dot additions", () => {
    const cases = [
      program("mul_add_goal", p => add(mul(
        p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
        p.argument("c", Tensor.f32([2])))),
      program("dot_add_goal", p => add(dot(
        p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
        p.argument("c", Tensor.f32([])))),
    ]
    for (const [caseIndex, source] of cases.entries()) {
      const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
      const executable = session.compile(source, { optimizationGoal: "memory" })
      const a = session.tensor([1, 2]), b = session.tensor([3, 4])
      const c = caseIndex === 0 ? session.tensor([5, 6]) : session.tensor(5)
      const result = executable.run({ a, b, c }) as any
      {
        const implementations = (session.runtimeEvidence(executable) as any).entries
          .map((entry: any) => entry.physical_implementation)
        expect(implementations.length >= 1).toBe(true)
        expect(implementations.every((id: string) =>
          id.startsWith("cpu_"))).toBe(true)
      }
      result.dispose(); a.dispose(); b.dispose(); c.dispose(); executable.dispose(); session.dispose()
    }
  })

  test("selects dense matrix epilogues", () => {
    for (const size of [8, 16, 128]) {
      const source = program(`dense_epilogue_goal_${size}`, p => add(
        matmul(p.argument("left", Tensor.f32([size, size])),
          p.argument("right", Tensor.f32([size, size]))),
        p.argument("bias", Tensor.f32([size])),
      ))
      const matrix = Array.from({ length: size }, (_, row) =>
        Array.from({ length: size }, (_, column) => row === column ? 1 : 0))
      for (const optimizationGoal of ["balanced", "latency", "throughput"] as const) {
        const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
        const executable = session.compile(source, { optimizationGoal, denseInputs: true })
        const left = session.tensor(matrix), right = session.tensor(matrix)
        const bias = session.tensor(Array(size).fill(1))
        const result = executable.run({ left, right, bias }) as any
        expect(result.to_array()).toEqual(matrix.map(row => row.map(value => value + 1)))
        {
          const actual = (session.runtimeEvidence(executable) as any).entries.map((entry: any) => entry.physical_implementation)
          expect(actual.length >= 1).toBe(true)
          expect(actual.every((id: string) => id.startsWith("cpu_"))).toBe(true)
        }
        result.dispose(); left.dispose(); right.dispose(); bias.dispose(); executable.dispose(); session.dispose()
      }
    }
  })

  test("reports the baseline plan when optimization is disabled", () => {
    const source = program("no_optimization", p => add(add(
      p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
      p.argument("c", Tensor.f32([2]))))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
    const executable = session.compile(source, { optimizationLevel: "none" })
    const a = session.tensor([1, 2]), b = session.tensor([3, 4]), c = session.tensor([5, 6])
    const result = executable.run({ a, b, c }) as any
    expect(result.to_array()).toEqual([9, 12])
    {
      const actual = (session.runtimeEvidence(executable) as any).entries.map((entry: any) => entry.physical_implementation)
      expect(actual).toEqual(["cpu_add", "cpu_add"])
    }
    result.dispose(); a.dispose(); b.dispose(); c.dispose(); executable.dispose(); session.dispose()
  })

  test("applies bounded search limits on the Compute Core compiler", () => {
    const source = program("bounded_search", p => add(add(
      p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
      p.argument("c", Tensor.f32([2]))))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 8 } })
    const executable = session.compile(source, { searchLimits: { structuralStates: 1 } })
    {
      const explanation = executable.explain() as any
      expect(explanation.search.limits.structural_states).toBe(1)
      expect(explanation.search.evaluated_states).toBe(1)
      expect(explanation.search.truncated).toBe(true)
      expect(explanation.selected_physical_nodes.map((node: any) => node.implementation))
        .toEqual(["cpu_add", "cpu_add"])
    }
    const a = session.tensor([1, 2]), b = session.tensor([3, 4]), c = session.tensor([5, 6])
    const result = executable.run({ a, b, c }) as any
    expect(result.to_array()).toEqual([9, 12])
    {
      const evidence = session.runtimeEvidence(executable) as any
      expect(evidence.schema_version).toBe(11)
      expect(evidence.search.structural_states).toBe(1)
      expect(evidence.search.evaluated_states).toBe(1)
      expect(evidence.search.truncated).toBe(true)
    }
    result.dispose(); a.dispose(); b.dispose(); c.dispose(); executable.dispose(); session.dispose()
  })

  test("applies every public search limit on the Compute Core compiler", () => {
    const source = program("all_search_limits", p => add(add(
      p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))),
      p.argument("c", Tensor.f32([2]))))
    const session = new Session({ device: "cpu" })
    const cases = [
      ["structuralStates", 1],
      ["structuralExpansions", 1],
      ["completeEvaluations", 1],
      ["physicalWork", 1],
      ["readyWork", 1],
    ] as const
    for (const [name, value] of cases) {
      const executable = session.compile(source, { searchLimits: { [name]: value } })
      {
        const search = (executable.explain() as any).search
        const field = name.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`)
        expect(search.limits[field]).toBe(value)
        expect(search.complete_evaluations >= 1).toBe(true)
      }
      executable.dispose()
    }
    session.dispose()
  })

  test("keeps residual-free programs on Compute Core under recompute policy", () => {
    const source = program("residual_policy", p => add(
      p.argument("a", Tensor.f32([2])), p.argument("b", Tensor.f32([2]))))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source, { residualPolicy: "recompute" })
    const a = session.tensor([1, 2]), b = session.tensor([3, 4])
    const result = executable.run({ a, b }) as any
    expect(result.to_array()).toEqual([4, 6])
    result.dispose(); a.dispose(); b.dispose(); executable.dispose(); session.dispose()
  })

  test("reuses a prepared graph while replacing changed input values", () => {
    const source = program("changed_input_plan", p => add(
      p.argument("x", Tensor.f32([2])), p.argument("bias", Tensor.f32([2])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source)
    const x = session.tensor([1, 2])
    const changed = session.tensor([3, 4])
    const bias = session.tensor([5, 6])
    for (const [value, expected] of [[x, [6, 8]], [x, [6, 8]], [changed, [8, 10]]] as const) {
      const result = executable.run({ x: value, bias }) as any
      expect(result.to_array()).toEqual(expected)
      result.dispose()
    }
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.execution_path).toBe("prepared")
      expect(evidence.prepared_cache_misses).toBe(1)
      expect(evidence.prepared_cache_hits).toBe(2)
      expect(evidence.observed_run_count).toBe(3)
      expect(evidence.entries[0].invocation_count).toBe(3)
    }
    x.dispose(); changed.dispose(); bias.dispose(); executable.dispose(); session.dispose()
  })

  test("reports Compute Core capture overflow", () => {
    const source = program("evidence_overflow", p => neg(p.argument("value", Tensor.f32([2]))))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 1 } })
    const executable = session.compile(source)
    const value = session.tensor([1, 2])
    session.resetEvidence()
    const result = executable.run({ value }) as any
    result.dispose()
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.dropped_event_count > 0).toBe(true)
      expect(evidence.run_count_complete).toBe(false)
      expect(evidence.kernel_count_evidence).toBe("unavailable")
    }
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("attributes Compute Core view execution to its semantic step", () => {
    const source = program("evidence_view", p => transpose(p.argument("value", Tensor.f32([2, 3])), [1, 0]))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source)
    const value = session.tensor([[1, 2, 3], [4, 5, 6]])
    session.resetEvidence()
    const result = executable.run({ value }) as any
    expect(result.to_array()).toEqual([[1, 4], [2, 5], [3, 6]])
    const evidence = session.runtimeEvidence(executable) as any
    {
      const entry = evidence.entries.find((item: any) => item.operation === "transpose")
      expect(entry.graph_kind).toBe("resident_view")
      expect(entry.invocation_count).toBe(1)
      expect(entry.materialized_view_count).toBe(0)
      expect(entry.kernel_count).toBe(0)
      expect(evidence.execution_path).toBe("resident_view")
    }
    result.dispose()
    value.dispose()
    executable.dispose()
    session.dispose()
  })

  test("records failed execution without a success completion", () => {
    const source = program("evidence_failure", p => div(
      p.argument("left", Tensor.i64([1])), p.argument("right", Tensor.i64([1])),
    ))
    const session = new Session({ device: "cpu", telemetry: { captureRecords: 64 } })
    const executable = session.compile(source)
    const left = session.tensor([7], { dtype: "i64" })
    const right = session.tensor([0], { dtype: "i64" })
    session.resetEvidence()
    expect(() => executable.run({ left, right })).toThrow()
    const evidence = session.runtimeEvidence(executable) as any
    {
      expect(evidence.observed_run_count).toBe(1)
      expect(evidence.observed_failed_run_count).toBe(1)
      expect(evidence.entries[0].failed_count).toBe(1)
    }
    left.dispose()
    right.dispose()
    executable.dispose()
    session.dispose()
  })

  test("returns public layout views from Compute Core without losing stride guards", () => {
    const source = program("public_layout_views", p => {
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
    const denseSource = program("layout_guard", p => sin(
      p.argument("x", Tensor.f32([3, 2])),
    ))
    const session = new Session({ device: "cpu" })
    const executable = session.compile(source)
    const dense = session.compile(denseSource, { denseInputs: true })

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

  test("executes indexed Programs through the in-process Compute Core Program", () => {
    const source = program("public_indexed_ops", p => {
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

  test("preserves indexed failures and reusable executables on Compute Core", () => {
    const selectedSource = program("public_index_select_bounds", p => index_select(
      p.argument("input", Tensor.f32([2, 2])),
      0,
      p.argument("indices", Tensor.i64([1])),
    ))
    const scatteredSource = program("public_scatter_bounds", p => scatter_add(
      p.argument("base", Tensor.f32([2, 2])),
      p.argument("indices", Tensor.i64([1, 2])),
      p.argument("updates", Tensor.f32([1, 2])),
      0,
    ))
    const session = new Session({ device: "cpu" })
    const selected = session.compile(selectedSource)
    const scattered = session.compile(scatteredSource)

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
