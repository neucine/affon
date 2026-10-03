import { create_inspection_model } from '../../packages/@affon/inspector/src/index.js'

const sample = Object.freeze({
  schema_version: 1, name: 'arithmetic_example', provenance: 'example', provenance_id: 'affon:example', kind: 'authored', constant_values: 'inline',
  arguments: [{ name: 'x', role: 'argument', spec: { dtype: 'f32', shape: [2], axes: ['feature'] }, provenance: 'arithmetic_example.x' }],
  parameters: [{ name: 'shift.bias', role: 'parameter', spec: { dtype: 'f32', shape: [2], axes: ['feature'] }, provenance: 'arithmetic_example.shift.bias' }],
  state: [], constants: [],
  nodes: [
    { id: 0, kind: 'argument', role: 'argument', name: 'x', provenance: 'arithmetic_example.x', path: [], spec: { dtype: 'f32', shape: [2], axes: ['feature'] } },
    { id: 1, kind: 'parameter', role: 'parameter', name: 'shift.bias', provenance: 'arithmetic_example.shift.bias', path: [{ program: 'shift', instance: 'shift' }], spec: { dtype: 'f32', shape: [2], axes: ['feature'] }, options: { initializer: { kind: 'zeros' } } },
    { id: 2, kind: 'operation', op: 'add', operands: [0, 1], path: [{ program: 'shift', instance: 'shift' }], spec: { dtype: 'f32', shape: [2], axes: ['feature'] } },
    { id: 3, kind: 'operation', op: 'mul', operands: [2, 2], path: [], spec: { dtype: 'f32', shape: [2], axes: ['feature'] } },
  ],
  components: [{
    id: 'component:shift@shift#0', program: 'shift', instance: 'shift', path: [{ program: 'shift', instance: 'shift' }],
    bindings: { value: 0 }, outputs: [2],
  }],
  outputs: [3], transitions: [],
})

const $ = selector => document.querySelector(selector)
const svg_ns = 'http://www.w3.org/2000/svg'
let model = create_inspection_model(sample)
let expanded = new Set(['scope:'])
let selection = { kind: 'scope', id: 'scope:' }
let search = { node_ids: [], scope_ids: [] }

function role_for(id) { return model.output_ids.has(id) ? 'output' : model.roles_by_id.get(id).role }
function node_label(node) { return node.name ?? node.op ?? `${node.kind} ${node.id}` }
function path_label(path) { return path.length ? path.map(segment => `${segment.instance} (${segment.program})`).join(' / ') : 'root' }
function el(tag, class_name, text) {
  const node = document.createElement(tag)
  if (class_name) node.className = class_name
  if (text !== undefined) node.textContent = text
  return node
}
function svg_el(tag, attributes = {}) {
  const node = document.createElementNS(svg_ns, tag)
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value))
  return node
}

function matches(kind, id, role) {
  const query = $('#search').value.trim()
  const role_filter = $('#role').value
  const search_match = !query || (kind === 'node' ? search.node_ids.includes(id) : search.scope_ids.includes(id))
  return search_match && (!role_filter || (kind === 'node' && role === role_filter))
}

function set_selection(kind, id) {
  selection = { kind, id }
  render_structure()
  render_graph()
  render_details()
}

function toggle_scope(id) {
  if (id === 'scope:') return set_selection('scope', id)
  if (expanded.has(id)) expanded.delete(id); else expanded.add(id)
  selection = { kind: 'scope', id }
  render_structure()
  render_graph()
  render_details()
}

function render_overview() {
  const inspection = model.inspection
  $('#program-name').textContent = inspection.name
  $('#program-kind').textContent = `${inspection.kind} · schema v${inspection.schema_version ?? 0} · ${inspection.nodes.length} nodes · serialized ProgramInspection`
  const counts = [
    ['arguments', inspection.arguments.length], ['parameters', inspection.parameters.length], ['state', inspection.state.length],
    ['constants', inspection.constants.length], ['outputs', inspection.outputs.length], ['transitions', inspection.transitions.length],
  ]
  $('#summary').replaceChildren(...counts.map(([label, count]) => el('span', 'chip', `${count} ${label}`)))
  const roles = [...new Set([...model.roles_by_id.values()].map(value => value.role).concat(['output']))]
  const current = $('#role').value
  $('#role').replaceChildren(new Option('All roles', ''), ...roles.sort().map(role => new Option(role, role)))
  $('#role').value = roles.includes(current) ? current : ''
}

function render_structure() {
  const container = $('#structure')
  container.replaceChildren()
  const visit = (scope_id, depth) => {
    const scope = model.scopes_by_id.get(scope_id)
    const open = expanded.has(scope_id)
    const row = el('div', `tree-row${selection.kind === 'scope' && selection.id === scope_id ? ' selected' : ''}`)
    row.style.paddingLeft = `${depth * 14}px`
    if (!matches('scope', scope_id)) row.classList.add('dim')
    const toggle = el('button', 'tree-toggle', scope_id === 'scope:' ? '⌂' : open ? '▾' : '▸')
    toggle.disabled = scope_id === 'scope:'
    toggle.addEventListener('click', event => {
      event.stopPropagation()
      toggle_scope(scope_id)
    })
    const select = el('button', 'tree-select', scope_id === 'scope:' ? model.inspection.name : scope.path.at(-1).instance)
    select.title = path_label(scope.path)
    select.addEventListener('click', () => toggle_scope(scope_id))
    row.append(toggle, select)
    container.append(row)
    if (!open) return
    for (const node_id of scope.node_ids) {
      const node = model.nodes_by_id.get(node_id), role = role_for(node_id)
      const node_row = el('div', `tree-row role-${role}${selection.kind === 'node' && selection.id === node_id ? ' selected' : ''}`)
      node_row.style.paddingLeft = `${(depth + 1) * 14 + 24}px`
      if (!matches('node', node_id, role)) node_row.classList.add('dim')
      node_row.append(el('span', 'role-dot'), el('span', 'node-id', String(node_id)))
      const button = el('button', 'tree-select', node_label(node))
      button.addEventListener('click', () => set_selection('node', node_id))
      node_row.append(button)
      container.append(node_row)
    }
    for (const child of scope.child_ids) visit(child, depth + 1)
  }
  visit('scope:', 0)
}

function graph_layout(graph) {
  const incoming = new Map(graph.nodes.map(node => [node.id, []]))
  const outgoing = new Map(graph.nodes.map(node => [node.id, []]))
  for (const edge of graph.edges) { incoming.get(edge.to).push(edge.from); outgoing.get(edge.from).push(edge.to) }
  const rank = new Map(graph.nodes.map(node => [node.id, 0]))
  const indegree = new Map([...incoming].map(([id, values]) => [id, values.length]))
  const queue = [...indegree].filter(([, degree]) => degree === 0).map(([id]) => id)
  while (queue.length) {
    const id = queue.shift()
    for (const next of outgoing.get(id)) {
      rank.set(next, Math.max(rank.get(next), rank.get(id) + 1))
      indegree.set(next, indegree.get(next) - 1)
      if (indegree.get(next) === 0) queue.push(next)
    }
  }
  const columns = new Map()
  for (const node of graph.nodes) {
    const column = columns.get(rank.get(node.id)) ?? []
    column.push(node); columns.set(rank.get(node.id), column)
  }
  const positions = new Map()
  for (const [column, nodes] of columns) nodes.forEach((node, row) => positions.set(node.id, { x: 28 + column * 190, y: 32 + row * 76 }))
  return { positions, width: Math.max(520, (Math.max(0, ...rank.values()) + 1) * 190 + 28), height: Math.max(430, Math.max(0, ...[...columns.values()].map(nodes => nodes.length)) * 76 + 38) }
}

function render_graph() {
  const svg = $('#graph svg'), graph = model.visible_graph(expanded), layout = graph_layout(graph)
  svg.replaceChildren()
  svg.setAttribute('viewBox', `0 0 ${layout.width} ${layout.height}`)
  svg.style.width = `${layout.width}px`; svg.style.height = `${layout.height}px`
  const defs = svg_el('defs'), marker = svg_el('marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' })
  marker.append(svg_el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#46515c' })); defs.append(marker); svg.append(defs)
  for (const edge of graph.edges) {
    const from = layout.positions.get(edge.from), to = layout.positions.get(edge.to)
    svg.append(svg_el('path', { class: 'edge', d: `M ${from.x + 146} ${from.y + 23} C ${from.x + 166} ${from.y + 23}, ${to.x - 20} ${to.y + 23}, ${to.x} ${to.y + 23}` }))
  }
  for (const node of graph.nodes) {
    const position = layout.positions.get(node.id), role = node.role
    const selected = node.kind === selection.kind && (node.kind === 'node' ? node.node_id === selection.id : node.scope_id === selection.id)
    const group = svg_el('g', {
      class: `graph-node ${node.kind} role-${role}${node.is_component_output ? ' component-output' : ''}${node.is_program_output ? ' program-output' : ''}${selected ? ' selected' : ''}`,
      transform: `translate(${position.x} ${position.y})`, tabindex: 0,
    })
    if (!matches(node.kind, node.kind === 'node' ? node.node_id : node.scope_id, role)) group.classList.add('dim')
    group.append(svg_el('rect', { width: 146, height: 46 }))
    const title = svg_el('text', { x: 10, y: 19 }), meta = svg_el('text', { x: 10, y: 35, class: 'meta' })
    title.textContent = node.label.length > 20 ? `${node.label.slice(0, 19)}…` : node.label
    const output_label = node.is_program_output
      ? 'program out'
      : node.is_component_output ? `${node.component_output_count > 1 ? `${node.component_output_count} ` : ''}component out` : ''
    meta.textContent = node.kind === 'node'
      ? [`node:${node.node_id}`, role, output_label].filter(Boolean).join(' · ')
      : [`${model.scope_node_ids(node.scope_id, { recursive: true }).length} nodes`, output_label || 'scope'].join(' · ')
    group.append(title, meta)
    if (node.is_component_output || node.is_program_output) {
      group.append(svg_el('circle', { class: 'output-port', cx: 146, cy: 23, r: node.is_program_output ? 6 : 5 }))
    }
    group.addEventListener('click', () => node.kind === 'scope'
      ? toggle_scope(node.scope_id)
      : set_selection('node', node.node_id))
    svg.append(group)
  }
}

function field(name, value, class_name = '') {
  const wrapper = el('dl', 'field'), term = el('dt', '', name), detail = el('dd', class_name, value)
  wrapper.append(term, detail); return wrapper
}

function links(name, ids) {
  const wrapper = el('dl', 'field'), term = el('dt', '', name), detail = el('dd', 'link-list')
  if (!ids.length) detail.textContent = 'none'
  for (const id of ids) {
    const button = el('button', 'node-link', `node:${id}`)
    button.addEventListener('click', () => set_selection('node', id)); detail.append(button)
  }
  wrapper.append(term, detail); return wrapper
}

function binding_links(bindings) {
  const wrapper = el('dl', 'field'), term = el('dt', '', 'Bindings'), detail = el('dd', 'link-list')
  for (const [name, id] of Object.entries(bindings)) {
    const button = el('button', 'node-link', `${name} → node:${id}`)
    button.addEventListener('click', () => set_selection('node', id)); detail.append(button)
  }
  if (!Object.keys(bindings).length) detail.textContent = 'none'
  wrapper.append(term, detail); return wrapper
}

function title(locator, label) {
  const wrapper = el('div', 'detail-title'), heading = el('h3', '', label), copy = el('button', 'copy', 'Copy locator')
  copy.addEventListener('click', async () => { await navigator.clipboard.writeText(locator); copy.textContent = 'Copied' })
  wrapper.append(heading, copy); return wrapper
}

function render_details() {
  const container = $('#details')
  container.replaceChildren()
  $('#selection-kind').textContent = selection.kind
  if (selection.kind === 'node') {
    const node = model.nodes_by_id.get(selection.id)
    if (!node) { container.append(el('p', 'empty', 'Selected node is unavailable.')); return }
    const role = model.roles_by_id.get(node.id)
    const component_outputs = model.component_outputs_by_node_id.get(node.id) ?? []
    container.append(
      title(`node:${node.id}`, node_label(node)),
      field('Locator', `node:${node.id}`, 'mono'),
      field('Role', `${role.role}${role.is_output ? ' · output' : ''}`),
      field('Specification', `${node.spec.dtype} [${node.spec.shape.join(', ')}]${node.spec.axes ? ` · axes: ${node.spec.axes.join(', ')}` : ''}`, 'mono'),
      field('Composition path', path_label(node.path)),
      links('Operands', node.operands ?? []),
      links('Consumers', model.consumers_by_id.get(node.id) ?? []),
    )
    if (component_outputs.length) container.append(field('Component output of', component_outputs.map(id => {
      const component = model.components_by_id.get(id)
      return component ? `${component.program} as ${component.instance}` : id
    }).join('\n'), 'mono'))
    if (role.is_output) container.append(field('Program output', 'Terminal for this inspected Program; it may still be a component output at an outer boundary'))
    if (node.provenance) container.append(field('Provenance', node.provenance, 'mono'))
    if (node.options) container.append(field('Options / initializer', JSON.stringify(node.options, null, 2), 'mono'))
    if (node.value !== undefined) container.append(field('Captured value', 'Present in inspection; hidden by default'))
    else if (node.value_summary) container.append(field('Constant value', `Summarized · ${node.value_summary.elements} elements`))
    else if (node.role === 'constant') container.append(field('Constant value', `Redacted by ${model.inspection.constant_values ?? 'legacy'} policy`))
    const transitions = model.transitions_by_parameter_id.get(node.id) ?? []
    if (transitions.length) container.append(field('Transitions', transitions.map(index => `${index}: ${model.inspection.transitions[index].kind}`).join('\n'), 'mono'))
  } else {
    const scope = model.scopes_by_id.get(selection.id)
    if (!scope) { container.append(el('p', 'empty', 'Selected scope is unavailable.')); return }
    const recursive = model.scope_node_ids(scope.id, { recursive: true })
    const role_counts = {}
    for (const id of recursive) { const role = role_for(id); role_counts[role] = (role_counts[role] ?? 0) + 1 }
    container.append(
      title(scope.id, scope.id === 'scope:' ? model.inspection.name : scope.path.at(-1).instance),
      field('Locator', scope.id, 'mono'),
      field('Composition path', path_label(scope.path)),
      field('Contained nodes', `${recursive.length} total · ${scope.node_ids.length} direct`),
      field('Roles', Object.entries(role_counts).map(([role, count]) => `${role}: ${count}`).join('\n'), 'mono'),
      links('Direct nodes', scope.node_ids),
    )
    for (const component_id of scope.component_ids) {
      const component = model.components_by_id.get(component_id)
      container.append(
        field('Component', `${component.program} as ${component.instance}`, 'mono'),
        field('Component ID', component.id, 'mono'),
        binding_links(component.bindings),
        links('Component outputs', component.outputs),
      )
    }
    if (scope.id === 'scope:' && model.inspection.provenance_id) container.append(field('Provenance identity', model.inspection.provenance_id, 'mono'))
  }
}

function render() { render_overview(); render_structure(); render_graph(); render_details() }

function open_inspection(inspection) {
  model = create_inspection_model(inspection)
  expanded = new Set(['scope:'])
  selection = { kind: 'scope', id: 'scope:' }
  $('#search').value = ''; $('#role').value = ''; search = { node_ids: [], scope_ids: [] }
  render()
}

async function open_file(file) {
  try { open_inspection(JSON.parse(await file.text())) }
  catch (error) { window.alert(`Could not open inspection: ${error.message}`) }
}

$('#file').addEventListener('change', event => { if (event.target.files[0]) open_file(event.target.files[0]); event.target.value = '' })
$('#search').addEventListener('input', event => { search = model.search(event.target.value); render_structure(); render_graph() })
$('#role').addEventListener('change', () => { render_structure(); render_graph() })
let drag_depth = 0
window.addEventListener('dragenter', event => { event.preventDefault(); drag_depth++; $('#drop-zone').hidden = false })
window.addEventListener('dragleave', () => { if (--drag_depth === 0) $('#drop-zone').hidden = true })
window.addEventListener('dragover', event => event.preventDefault())
window.addEventListener('drop', event => { event.preventDefault(); drag_depth = 0; $('#drop-zone').hidden = true; if (event.dataTransfer.files[0]) open_file(event.dataTransfer.files[0]) })
render()
