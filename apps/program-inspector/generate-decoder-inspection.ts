import fs from 'std:fs'
import { Tensor, program } from 'affon:compute'
import { prepare_inspection_export } from '@affon/inspector'
import { DecoderModel } from '../decoder-lm/src/model.ts'

const decoder = DecoderModel(32, 16, {
  numLayers: 2,
  numHeads: 4,
  hiddenDim: 32,
  causal: true,
  positional: 'learned',
  maxSeqLen: 16,
  tieEmbeddings: true,
})

const source = program('decoder_lm_inspection', p => decoder({
  token_ids: p.argument('token_ids', Tensor.i64([2, 8], { axes: ['batch', 'token'] })),
}, 'decoder'))

const inspection = prepare_inspection_export(source.inspect(), { constants: 'summary' })
const output = '/private/tmp/decoder-lm.affon-inspection.json'
fs.writeFileSync(output, JSON.stringify(inspection))
console.log(JSON.stringify({
  output,
  nodes: inspection.nodes.length,
  components: inspection.components.length,
  parameters: inspection.parameters.length,
  outputs: inspection.outputs.length,
}))
