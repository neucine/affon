import { describe, expect, test } from 'std:test'
import { buildRoutingReport, type Routing, type RoutingEntry } from './route-report.ts'

function route(
  backend: Routing['backend'],
  category: Routing['category'],
  reason: string,
  operation: string | null = null,
  option: string | null = null,
): Routing {
  return {
    schema_version: 1,
    backend,
    configured_backend: 'cpu',
    category,
    reason,
    operation,
    option,
  }
}

function entry(name: string, routing: Routing): RoutingEntry {
  return { name, routing }
}

const cyna = route('cyna', 'success', 'lowered')
const operation = route('zig', 'operation', 'unsupported', 'unsupported_op', 'shape')
const observability = route('zig', 'observability', 'evidence_capture')

describe('Cyna routing report', () => {
  test('groups identical fallback routes and counts the suite', () => {
    const report = buildRoutingReport([
      entry('z_program', operation),
      entry('observability', observability),
      entry('a_program', operation),
      entry('cyna_program', cyna),
    ])

    expect(report.programs).toEqual({ total: 4, cyna: 1, zig: 3, cyna_percent: 25 })
    expect(report.fallback_groups).toHaveLength(2)

    const grouped = report.fallback_groups.find(group => group.category === 'operation')
    expect(grouped).toEqual({
      classification: 'portable_lowering',
      category: 'operation',
      reason: 'unsupported',
      operation: 'unsupported_op',
      option: 'shape',
      program_count: 2,
      programs: ['a_program', 'z_program'],
    })
    expect(report.highest_impact_portable_lowering).toEqual(grouped)
  })

  test('sorts groups and Programs deterministically', () => {
    const sameCount = [
      entry('cat_z', route('zig', 'backend', 'non_cpu_backend')),
      entry('obs', observability),
      entry('cat_a', route('zig', 'configuration', 'cyna_disabled')),
    ]
    const forward = buildRoutingReport(sameCount)
    const reverse = buildRoutingReport([...sameCount].reverse())

    expect(reverse).toEqual(forward)
    expect(forward.fallback_groups.map(group => group.reason)).toEqual([
      'non_cpu_backend',
      'cyna_disabled',
      'evidence_capture',
    ])
  })

  test('keeps non-lowering limits out of the next-target selection', () => {
    const report = buildRoutingReport([
      entry('generic_lowering', route('zig', 'lowering', 'allocation_failed')),
      entry('metal', route('zig', 'backend', 'non_cpu_backend')),
      entry('disabled', route('zig', 'configuration', 'cyna_disabled')),
    ])

    expect(report.fallback_groups.every(group => group.classification !== 'portable_lowering')).toBe(true)
    expect(report.fallback_groups.find(group => group.category === 'lowering')?.classification).toBe('unclassified')
    expect(report.highest_impact_portable_lowering).toEqual(null)
  })

  test('ranks the largest portable lowering group first', () => {
    const dense = route('zig', 'dense_graph', 'DenseGraphError', 'mul')
    const report = buildRoutingReport([
      entry('dense_a', dense),
      entry('dense_b', dense),
      entry('operation_a', operation),
      entry('operation_b', operation),
      entry('operation_c', route('zig', 'operation', 'unsupported', 'other_op')),
    ])

    expect(report.highest_impact_portable_lowering?.program_count).toBe(2)
    expect(report.highest_impact_portable_lowering?.reason).toBe('DenseGraphError')
  })

  test('reports an empty suite as fully unmeasured', () => {
    const report = buildRoutingReport([])
    expect(report.programs).toEqual({ total: 0, cyna: 0, zig: 0, cyna_percent: 0 })
    expect(report.fallback_groups).toEqual([])
    expect(report.highest_impact_portable_lowering).toEqual(null)
  })
})
