/** Inspector-side projections over core inspection contracts. */
import type { ProgramComponentInspection, ProgramInspection, ProgramPath } from 'affon:compute'

export type InspectionNode = ProgramInspection['nodes'][number]
export type InspectionRole = 'argument' | 'parameter' | 'state' | 'constant' | 'operation' | 'composition' | 'gradient'

export interface InspectionScope {
  readonly id: string
  readonly path: ProgramPath
  readonly parent_id: string | null
  readonly child_ids: readonly string[]
  readonly node_ids: readonly number[]
  readonly parameter_ids: readonly number[]
  readonly state_ids: readonly number[]
  readonly constant_ids: readonly number[]
  readonly component_ids: readonly string[]
}

export interface InspectionGraphNode {
  readonly id: string
  readonly kind: 'scope' | 'node'
  readonly node_id?: number
  readonly scope_id: string
  readonly label: string
  readonly role: InspectionRole | 'scope' | 'output'
  /** Backward-compatible alias for is_program_output. */
  readonly is_output: boolean
  readonly is_program_output: boolean
  readonly is_component_output: boolean
  readonly component_output_count: number
}

export interface InspectionGraph {
  readonly nodes: readonly InspectionGraphNode[]
  readonly edges: readonly Readonly<{ id: string; from: string; to: string }>[]
}

export interface InspectionModel {
  readonly inspection: ProgramInspection
  readonly nodes_by_id: ReadonlyMap<number, InspectionNode>
  readonly consumers_by_id: ReadonlyMap<number, readonly number[]>
  readonly scopes_by_id: ReadonlyMap<string, InspectionScope>
  readonly components_by_id: ReadonlyMap<string, ProgramComponentInspection>
  readonly components_by_scope_id: ReadonlyMap<string, readonly string[]>
  /** Component interfaces that expose each node as an output. */
  readonly component_outputs_by_node_id: ReadonlyMap<number, readonly string[]>
  readonly root_scope: InspectionScope
  readonly output_ids: ReadonlySet<number>
  readonly roles_by_id: ReadonlyMap<number, Readonly<{ role: InspectionRole; is_output: boolean }>>
  readonly node_scope_ids: ReadonlyMap<number, string>
  readonly transitions_by_parameter_id: ReadonlyMap<number, readonly number[]>
  scope_node_ids(id: string, options?: { recursive?: boolean }): readonly number[]
  trace_upstream(id: number): readonly number[]
  trace_downstream(id: number): readonly number[]
  search(query: string): Readonly<{ node_ids: readonly number[]; scope_ids: readonly string[] }>
  visible_graph(expanded_scope_ids?: Iterable<string>): InspectionGraph
}

/** Return the stable textual locator for a composition path. */
export function scope_id(path: ProgramPath): string

/** Prepare versioned inspection JSON with constants inline, summarized, or redacted. */
export function prepare_inspection_export(
  inspection: ProgramInspection,
  options?: { constants?: 'inline' | 'summary' | 'redacted' },
): ProgramInspection

/** Build an immutable, DOM-free lookup and graph projection over serialized inspection data. */
export function create_inspection_model(inspection: ProgramInspection): InspectionModel
