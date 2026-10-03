// Inspector-side projections over core inspection contracts.
function readonly_map(source) {
  let result
  result = {
    get size() { return source.size },
    get(key) { return source.get(key) },
    has(key) { return source.has(key) },
    entries() { return source.entries() },
    keys() { return source.keys() },
    values() { return source.values() },
    forEach(callback, this_arg) {
      source.forEach((value, key) => callback.call(this_arg, value, key, result))
    },
    [Symbol.iterator]() { return source[Symbol.iterator]() },
  }
  return Object.freeze(result)
}

function readonly_set(source) {
  let result
  result = {
    get size() { return source.size },
    has(value) { return source.has(value) },
    entries() { return source.entries() },
    keys() { return source.keys() },
    values() { return source.values() },
    forEach(callback, this_arg) {
      source.forEach(value => callback.call(this_arg, value, value, result))
    },
    [Symbol.iterator]() { return source[Symbol.iterator]() },
  }
  return Object.freeze(result)
}

function freeze_array(values) { return Object.freeze([...values]) }
function freeze_path(path) {
  return Object.freeze(path.map(segment => Object.freeze({ program: segment.program, instance: segment.instance })))
}

function assert_inspection(inspection) {
  if (!inspection || typeof inspection !== 'object') throw new TypeError('Program inspection must be an object')
  if (typeof inspection.name !== 'string' || !inspection.name) throw new TypeError('Program inspection requires a name')
  if (!Array.isArray(inspection.nodes)) throw new TypeError('Program inspection nodes must be an array')
  if (!Array.isArray(inspection.outputs)) throw new TypeError('Program inspection outputs must be an array')
  if (!Array.isArray(inspection.transitions)) throw new TypeError('Program inspection transitions must be an array')
  if (inspection.schema_version !== undefined && inspection.schema_version !== 1) throw new TypeError(`Unsupported Program inspection schema version: ${inspection.schema_version}`)
}

/** Return a stable textual ID for one Program composition path. */
export function scope_id(path) {
  if (!path.length) return 'scope:'
  return `scope:${path.map(segment => `${encodeURIComponent(segment.program)}@${encodeURIComponent(segment.instance)}`).join('/')}`
}

function node_role(node) {
  if (node.role) return node.role
  if (node.kind === 'gradient' || node.kind === 'composition') return node.kind
  return 'operation'
}

function node_label(node) {
  return node.name ?? node.op ?? `${node.kind} ${node.id}`
}

function searchable_node_text(node) {
  return [
    node.id,
    node.kind,
    node.role,
    node.name,
    node.op,
    node.provenance,
    node.spec?.dtype,
    ...(node.spec?.shape ?? []),
    ...(node.spec?.axes ?? []),
    ...(node.path ?? []).flatMap(segment => [segment.program, segment.instance]),
  ].filter(value => value !== undefined).join(' ').toLowerCase()
}

function searchable_scope_text(scope) {
  return [scope.id, ...scope.path.flatMap(segment => [segment.program, segment.instance])].join(' ').toLowerCase()
}

function compact_identity(value) {
  let first = 0x811c9dc5, second = 0x9e3779b9
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index)
    first = Math.imul(first ^ code, 0x01000193)
    second = Math.imul(second ^ (code + index), 0x85ebca6b)
  }
  return `affon:${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0).toString(16).padStart(8, '0')}:${value.length}`
}

/** Create a versioned, JSON-ready inspection with an explicit constant policy. */
export function prepare_inspection_export(inspection, options = {}) {
  assert_inspection(inspection)
  const constants = options.constants ?? 'summary'
  if (constants !== 'inline' && constants !== 'summary' && constants !== 'redacted') {
    throw new TypeError('Inspection export constants must be inline, summary, or redacted')
  }
  const nodes = Object.freeze(inspection.nodes.map(node => {
    if (node.role !== 'constant' || constants === 'inline') return node
    const { value, value_summary, ...metadata } = node
    if (constants === 'summary') return Object.freeze({ ...metadata, value_summary: Object.freeze({ elements: node.spec.shape.reduce((product, size) => product * size, 1) }) })
    return Object.freeze(metadata)
  }))
  const parameter_ids_by_name = new Map(nodes.filter(node => node.role === 'parameter').flatMap(node => [
    ...(node.name ? [[node.name, node.id]] : []),
    ...(node.provenance ? [[node.provenance, node.id]] : []),
  ]))
  const transitions = Object.freeze(inspection.transitions.map(transition => Object.freeze({
    ...transition,
    parameters: freeze_array(transition.parameters ?? []),
    parameter_ids: freeze_array(transition.parameter_ids ?? (transition.parameters ?? []).map(name => parameter_ids_by_name.get(name)).filter(id => id !== undefined)),
  })))
  return Object.freeze({
    ...inspection,
    schema_version: 1,
    provenance_id: inspection.provenance_id ?? compact_identity(inspection.provenance ?? inspection.name),
    constant_values: constants,
    nodes,
    components: freeze_array(inspection.components ?? []),
    outputs: freeze_array(inspection.outputs),
    transitions,
  })
}

/**
 * Normalize and index serialized ProgramInspection data without depending on a
 * DOM, Program instance, or Session.
 */
export function create_inspection_model(inspection) {
  assert_inspection(inspection)

  const nodes = new Map()
  for (const node of inspection.nodes) {
    if (!Number.isSafeInteger(node?.id)) throw new TypeError('Every inspection node requires an integer id')
    if (nodes.has(node.id)) throw new TypeError(`Duplicate inspection node id: ${node.id}`)
    if (!Array.isArray(node.path)) throw new TypeError(`Inspection node ${node.id} requires a path`)
    nodes.set(node.id, node)
  }
  for (const node of nodes.values()) {
    for (const operand of node.operands ?? []) {
      if (!nodes.has(operand)) throw new TypeError(`Inspection node ${node.id} references missing operand ${operand}`)
    }
  }
  const outputs = new Set()
  for (const output of inspection.outputs) {
    if (!nodes.has(output)) throw new TypeError(`Inspection output references missing node ${output}`)
    outputs.add(output)
  }

  const consumers = new Map([...nodes.keys()].map(id => [id, []]))
  for (const node of nodes.values()) for (const operand of node.operands ?? []) {
    const ids = consumers.get(operand)
    if (!ids.includes(node.id)) ids.push(node.id)
  }

  const mutable_scopes = new Map()
  const ensure_scope = path => {
    const id = scope_id(path)
    if (mutable_scopes.has(id)) return mutable_scopes.get(id)
    const parent_path = path.slice(0, -1)
    const scope = {
      id,
      path: freeze_path(path),
      parent_id: path.length ? scope_id(parent_path) : null,
      child_ids: [],
      node_ids: [],
      parameter_ids: [],
      state_ids: [],
      constant_ids: [],
      component_ids: [],
    }
    mutable_scopes.set(id, scope)
    if (path.length) ensure_scope(parent_path).child_ids.push(id)
    return scope
  }
  ensure_scope([])

  const node_scope_ids = new Map()
  const roles = new Map()
  for (const node of nodes.values()) {
    for (let length = 1; length <= node.path.length; length++) ensure_scope(node.path.slice(0, length))
    const scope = ensure_scope(node.path)
    scope.node_ids.push(node.id)
    if (node.role === 'parameter') scope.parameter_ids.push(node.id)
    if (node.role === 'state') scope.state_ids.push(node.id)
    if (node.role === 'constant') scope.constant_ids.push(node.id)
    node_scope_ids.set(node.id, scope.id)
    roles.set(node.id, Object.freeze({ role: node_role(node), is_output: outputs.has(node.id) }))
  }

  const components = new Map()
  const components_by_scope_id = new Map()
  const component_outputs_by_node_id = new Map([...nodes.keys()].map(id => [id, []]))
  for (const component of inspection.components ?? []) {
    if (!component || typeof component.id !== 'string' || !component.id) throw new TypeError('Every inspection component requires an id')
    if (components.has(component.id)) throw new TypeError(`Duplicate inspection component id: ${component.id}`)
    if (!Array.isArray(component.path)) throw new TypeError(`Inspection component ${component.id} requires a path`)
    for (const [name, id] of Object.entries(component.bindings ?? {})) {
      if (!nodes.has(id)) throw new TypeError(`Inspection component ${component.id} binding ${name} references missing node ${id}`)
    }
    for (const id of component.outputs ?? []) if (!nodes.has(id)) throw new TypeError(`Inspection component ${component.id} output references missing node ${id}`)
    const normalized = Object.freeze({
      ...component,
      path: freeze_path(component.path),
      bindings: Object.freeze({ ...component.bindings }),
      outputs: freeze_array(component.outputs),
    })
    components.set(component.id, normalized)
    const scope = ensure_scope(normalized.path)
    scope.component_ids.push(component.id)
    const ids = components_by_scope_id.get(scope.id) ?? []
    ids.push(component.id)
    components_by_scope_id.set(scope.id, ids)
    for (const output of normalized.outputs) component_outputs_by_node_id.get(output).push(component.id)
  }

  const scopes = new Map()
  for (const [id, scope] of mutable_scopes) scopes.set(id, Object.freeze({
    ...scope,
    child_ids: freeze_array(scope.child_ids),
    node_ids: freeze_array(scope.node_ids),
    parameter_ids: freeze_array(scope.parameter_ids),
    state_ids: freeze_array(scope.state_ids),
    constant_ids: freeze_array(scope.constant_ids),
    component_ids: freeze_array(scope.component_ids),
  }))

  const transitions_by_parameter_id = new Map()
  const parameters_by_name = new Map([...nodes.values()].filter(node => node.role === 'parameter').flatMap(node => [
    ...(node.name ? [[node.name, node.id]] : []),
    ...(node.provenance ? [[node.provenance, node.id]] : []),
  ]))
  inspection.transitions.forEach((transition, index) => {
    const parameter_ids = Array.isArray(transition.parameter_ids)
      ? transition.parameter_ids
      : (transition.parameters ?? []).map(name => parameters_by_name.get(name)).filter(id => id !== undefined)
    for (const parameter_id of parameter_ids) {
      if (!nodes.has(parameter_id)) throw new TypeError(`Inspection transition ${index} references missing parameter node ${parameter_id}`)
      const indices = transitions_by_parameter_id.get(parameter_id) ?? []
      indices.push(index)
      transitions_by_parameter_id.set(parameter_id, indices)
    }
  })

  const frozen_consumers = new Map([...consumers].map(([id, ids]) => [id, freeze_array(ids)]))
  const frozen_transitions = new Map([...transitions_by_parameter_id].map(([id, indices]) => [id, freeze_array(indices)]))
  const frozen_components_by_scope = new Map([...components_by_scope_id].map(([id, component_ids]) => [id, freeze_array(component_ids)]))
  const frozen_component_outputs = new Map([...component_outputs_by_node_id].map(([id, component_ids]) => [id, freeze_array(component_ids)]))

  const scope_node_ids = (id, options = {}) => {
    const scope = scopes.get(id)
    if (!scope) return Object.freeze([])
    const result = [...scope.node_ids]
    if (options.recursive) for (const child of scope.child_ids) result.push(...scope_node_ids(child, options))
    return freeze_array(result)
  }

  const trace = (start, direction) => {
    if (!nodes.has(start)) return Object.freeze([])
    const seen = new Set()
    const pending = [start]
    while (pending.length) {
      const id = pending.pop()
      const adjacent = direction === 'upstream' ? (nodes.get(id).operands ?? []) : frozen_consumers.get(id)
      for (const next of adjacent) if (!seen.has(next) && next !== start) { seen.add(next); pending.push(next) }
    }
    return freeze_array(seen)
  }

  const model = {
    inspection,
    nodes_by_id: readonly_map(nodes),
    consumers_by_id: readonly_map(frozen_consumers),
    scopes_by_id: readonly_map(scopes),
    components_by_id: readonly_map(components),
    components_by_scope_id: readonly_map(frozen_components_by_scope),
    component_outputs_by_node_id: readonly_map(frozen_component_outputs),
    root_scope: scopes.get('scope:'),
    output_ids: readonly_set(outputs),
    roles_by_id: readonly_map(roles),
    node_scope_ids: readonly_map(node_scope_ids),
    transitions_by_parameter_id: readonly_map(frozen_transitions),
    scope_node_ids,
    trace_upstream(id) { return trace(id, 'upstream') },
    trace_downstream(id) { return trace(id, 'downstream') },
    search(query) {
      const needle = String(query ?? '').trim().toLowerCase()
      if (!needle) return Object.freeze({ node_ids: Object.freeze([]), scope_ids: Object.freeze([]) })
      return Object.freeze({
        node_ids: freeze_array([...nodes.values()].filter(node => searchable_node_text(node).includes(needle)).map(node => node.id)),
        scope_ids: freeze_array([...scopes.values()].filter(scope => searchable_scope_text(scope).includes(needle)).map(scope => scope.id)),
      })
    },
    visible_graph(expanded_scope_ids = []) {
      const expanded = new Set(expanded_scope_ids)
      expanded.add('scope:')
      const graph_nodes = []
      const visible_ids = new Set()
      const visit = id => {
        const scope = scopes.get(id)
        if (id !== 'scope:' && !expanded.has(id)) {
          const component_output_node_ids = new Set(scope.component_ids.flatMap(component_id => components.get(component_id)?.outputs ?? []))
          const is_program_output = scope_node_ids(id, { recursive: true }).some(node_id => outputs.has(node_id))
          graph_nodes.push(Object.freeze({
            id, kind: 'scope', scope_id: id, label: scope.path.at(-1)?.instance ?? inspection.name, role: 'scope',
            is_output: is_program_output, is_program_output,
            is_component_output: component_output_node_ids.size > 0,
            component_output_count: component_output_node_ids.size,
          }))
          visible_ids.add(id)
          return
        }
        for (const node_id of scope.node_ids) {
          const node = nodes.get(node_id)
          const id = `node:${node_id}`
          const is_program_output = outputs.has(node_id)
          const component_output_count = frozen_component_outputs.get(node_id).length
          graph_nodes.push(Object.freeze({
            id, kind: 'node', node_id, scope_id: scope.id, label: node_label(node),
            role: is_program_output ? 'output' : node_role(node), is_output: is_program_output, is_program_output,
            is_component_output: component_output_count > 0, component_output_count,
          }))
          visible_ids.add(id)
        }
        for (const child of scope.child_ids) visit(child)
      }
      visit('scope:')

      const endpoint = node_id => {
        const node = nodes.get(node_id)
        for (let length = 1; length <= node.path.length; length++) {
          const id = scope_id(node.path.slice(0, length))
          if (!expanded.has(id)) return id
        }
        return `node:${node_id}`
      }
      const edges = []
      const edge_ids = new Set()
      for (const node of nodes.values()) for (const operand of node.operands ?? []) {
        const from = endpoint(operand), to = endpoint(node.id), id = `${from}->${to}`
        if (from !== to && visible_ids.has(from) && visible_ids.has(to) && !edge_ids.has(id)) {
          edge_ids.add(id)
          edges.push(Object.freeze({ id, from, to }))
        }
      }
      return Object.freeze({ nodes: freeze_array(graph_nodes), edges: freeze_array(edges) })
    },
  }
  return Object.freeze(model)
}
