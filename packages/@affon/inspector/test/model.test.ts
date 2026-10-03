import { describe, expect, test } from 'std:test'
import { Tensor, program } from 'affon:compute'
import { add, mul } from 'affon:ops'
import { create_inspection_model, prepare_inspection_export, scope_id } from '@affon/inspector'
import { DecoderModel } from '../../../../apps/decoder-lm/src/model.ts'
import { load_graph } from '../../onnx/src/index.ts'

describe('@affon/inspector', () => {
  test('indexes a small arithmetic Program and derives immutable graph projections', () => {
    const child = program('arithmetic_child', p => add(
      p.argument('value', Tensor.f32([2], { axes: ['feature'] })),
      p.parameter('bias', Tensor.f32([2], { axes: ['feature'] })),
    ))
    const source = program('arithmetic', p => {
      const input = p.argument('input', Tensor.f32([2], { axes: ['feature'] }))
      const shifted = child({ value: input }, 'shift')
      return mul(shifted, shifted)
    })
    const model = create_inspection_model(source.inspect())
    const output = source.inspect().outputs[0]
    const shifted = source.inspect().nodes[output].operands![0]
    const child_scope = scope_id([{ program: 'arithmetic_child', instance: 'shift' }])

    expect(model.nodes_by_id.size).toBe(source.inspect().nodes.length)
    expect(model.root_scope.child_ids).toEqual([child_scope])
    expect(model.scopes_by_id.get(child_scope)!.parameter_ids.length).toBe(1)
    expect(model.scopes_by_id.get(child_scope)!.component_ids.length).toBe(1)
    expect(model.components_by_id.get(model.scopes_by_id.get(child_scope)!.component_ids[0])!.bindings).toEqual({ value: 0 })
    expect(model.component_outputs_by_node_id.get(shifted)).toEqual([model.scopes_by_id.get(child_scope)!.component_ids[0]])
    expect(model.consumers_by_id.get(shifted)).toEqual([output])
    expect(model.roles_by_id.get(output)).toEqual({ role: 'operation', is_output: true })
    expect(model.trace_upstream(output).length).toBe(3)
    expect(model.trace_downstream(shifted)).toEqual([output])
    expect(model.search('feature').node_ids.length > 0).toBe(true)
    expect(Object.isFrozen(model.consumers_by_id.get(shifted))).toBe(true)

    const collapsed = model.visible_graph()
    expect(collapsed.nodes.some(node => node.id === child_scope)).toBe(true)
    const collapsed_child = collapsed.nodes.find(node => node.id === child_scope)!
    expect([collapsed_child.is_component_output, collapsed_child.is_program_output, collapsed_child.component_output_count]).toEqual([true, false, 1])
    expect(collapsed.nodes.some(node => node.id === `node:${output}`)).toBe(true)
    expect(collapsed.edges.some(edge => edge.from === child_scope && edge.to === `node:${output}`)).toBe(true)
    const expanded = model.visible_graph([child_scope])
    expect(expanded.nodes.some(node => node.kind === 'scope')).toBe(false)
    const expanded_child_output = expanded.nodes.find(node => node.id === `node:${shifted}`)!
    expect([expanded_child_output.is_component_output, expanded_child_output.is_program_output]).toEqual([true, false])
  })

  test('indexes decoder composition paths and shared parameter consumers without model conventions', () => {
    const decoder = DecoderModel(16, 8, { numLayers: 1, numHeads: 2, hiddenDim: 16, causal: true, positional: 'learned', maxSeqLen: 8, tieEmbeddings: true })
    const source = program('inspection_decoder', p => decoder({
      token_ids: p.argument('token_ids', Tensor.i64([2, 3], { axes: ['batch', 'token'] })),
    }, 'decoder'))
    const model = create_inspection_model(source.inspect())
    const token_table = source.inspect().nodes.find(node => node.name === 'decoder.token_embedding.weight')!
    const attention_scope = scope_id([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
      { program: 'decoder_self_attention', instance: 'attention' },
    ])
    const block_scope = scope_id([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
    ])
    const attention_core_scope = scope_id([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
      { program: 'decoder_self_attention', instance: 'attention' },
      { program: 'decoder_attention_core', instance: 'core' },
    ])
    const feed_forward_scope = scope_id([
      { program: 'decoder_model', instance: 'decoder' },
      { program: 'decoder_block', instance: 'blocks.0' },
      { program: 'decoder_feed_forward', instance: 'feed_forward' },
    ])

    expect(model.scopes_by_id.has(attention_scope)).toBe(true)
    expect(model.scopes_by_id.has(attention_core_scope)).toBe(true)
    expect(model.scopes_by_id.has(feed_forward_scope)).toBe(true)
    expect(model.scopes_by_id.get(block_scope)!.node_ids.map(id => model.nodes_by_id.get(id)!.op)).toEqual(['add', 'add'])
    expect(model.scopes_by_id.get(block_scope)!.child_ids).toEqual([attention_scope, feed_forward_scope])
    const expanded_block = model.visible_graph([scope_id([{ program: 'decoder_model', instance: 'decoder' }]), block_scope])
    expect(expanded_block.nodes.filter(node => node.kind === 'scope').map(node => node.scope_id)).toContain(attention_scope)
    expect(expanded_block.nodes.filter(node => node.kind === 'scope').map(node => node.scope_id)).toContain(feed_forward_scope)
    expect(model.consumers_by_id.get(token_table.id)!.length).toBe(2)
    expect(model.search('decoder_attention_core').scope_ids).toEqual([attention_core_scope])
    expect(model.output_ids.size).toBe(1)
  })

  test('indexes an imported ONNX Program using only the generic inspection contract', () => {
    const imported = load_graph('packages/@affon/onnx/test/fixtures')
    try {
      const model = create_inspection_model(imported.forward.inspect())
      expect(model.inspection.name).toBe('onnx_import')
      expect(model.root_scope.child_ids).toEqual([])
      expect(model.root_scope.parameter_ids.length).toBe(Object.keys(imported.graph.constants).length)
      expect(model.output_ids.size).toBe(imported.graph.outputs.length)
      expect(model.search('matmul').node_ids.length > 0).toBe(true)
    } finally {
      for (const tensor of Object.values(imported.parameters)) tensor.dispose()
    }
  })

  test('rejects malformed serialized references at the adapter boundary', () => {
    const invalid = {
      name: 'invalid', provenance: '', kind: 'authored', arguments: [], parameters: [], state: [], constants: [],
      nodes: [{ id: 4, kind: 'operation', path: [], spec: { dtype: 'f32', shape: [1] }, op: 'add', operands: [3] }],
      outputs: [4], transitions: [],
    }
    expect(() => create_inspection_model(invalid as any)).toThrow('missing operand 3')
    expect(() => create_inspection_model({ ...invalid, schema_version: 2 } as any)).toThrow('Unsupported Program inspection schema version')
  })

  test('normalizes legacy JSON and prepares explicit constant export policies', () => {
    const source = program('constant_export', p => add(add(
      p.argument('x', Tensor.f32([2])),
      p.constant('offset', [1, 2], Tensor.f32([2])),
    ), p.parameter('weight', Tensor.f32([2]))))
    const inline = prepare_inspection_export(source.inspect(), { constants: 'inline' })
    const summary = prepare_inspection_export(source.inspect())
    const redacted = prepare_inspection_export(source.inspect(), { constants: 'redacted' })
    const constant_id = source.inspect().nodes.find(node => node.role === 'constant')!.id
    expect(inline.nodes.find(node => node.id === constant_id)!.value).toEqual([1, 2])
    expect(summary.constant_values).toBe('summary')
    expect(summary.nodes.find(node => node.id === constant_id)!.value).toBe(undefined)
    expect(summary.nodes.find(node => node.id === constant_id)!.value_summary).toEqual({ elements: 2 })
    expect(redacted.nodes.find(node => node.id === constant_id)!.value).toBe(undefined)
    expect(redacted.nodes.find(node => node.id === constant_id)!.value_summary).toBe(undefined)

    const legacy = JSON.parse(JSON.stringify(source.inspect()))
    delete legacy.schema_version
    delete legacy.provenance_id
    delete legacy.constant_values
    delete legacy.components
    legacy.transitions = [{ kind: 'optimize', optimizer: { kind: 'sgd', learning_rate: 0.1, momentum: 0 }, parameters: ['constant_export.weight'] }]
    const legacyModel = create_inspection_model(legacy)
    expect(legacyModel.nodes_by_id.size).toBe(source.inspect().nodes.length)
    const parameter_id = source.inspect().nodes.find(node => node.role === 'parameter')!.id
    expect(legacyModel.transitions_by_parameter_id.get(parameter_id)).toEqual([0])
    expect(prepare_inspection_export(legacy).transitions[0].parameter_ids).toEqual([parameter_id])
  })
})
