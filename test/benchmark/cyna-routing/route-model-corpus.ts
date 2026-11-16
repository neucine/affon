import { Session, Tensor } from 'affon:core'
import { load_graph } from '../../../packages/@affon/onnx/src/runtime.ts'
import { create_bert } from '../../../packages/@affon/models/src/bert/model.ts'
import { create_gpt2 } from '../../../packages/@affon/models/src/gpt2/model.ts'
import { create_llama } from '../../../packages/@affon/models/src/llama/model.ts'
import { create_vit } from '../../../packages/@affon/models/src/vit/model.ts'
import type { RegistryEntry } from './route-representative.ts'

type ModelDType = 'f32' | 'bf16'

function nested(shape: readonly number[], fill: number): any {
  return shape.length
    ? Array.from({ length: shape[0] }, () => nested(shape.slice(1), fill))
    : fill
}

function withModelSession<T>(dtype: ModelDType, author: (source: Session, value: (shape: readonly number[], fill?: number) => Tensor) => T): T {
  const source = new Session({ device: 'cpu' })
  try {
    return author(source, (shape, fill = 0.1) => source.tensor(nested(shape, fill), { dtype }) as Tensor)
  } finally {
    source.dispose()
  }
}

function imported(name: string, directory: string): RegistryEntry {
  const model = load_graph(directory)
  for (const parameter of Object.values(model.parameters)) parameter.dispose()
  return { name, corpus: 'imported', source: model.forward }
}

function gpt2(dtype: ModelDType, name: string): RegistryEntry {
  const source = withModelSession(dtype, (_session, value) => {
    const affine = (input: number, output: number) => ({
      weight: value([input, output]),
      bias: value([output], 0),
    })
    const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
    const model = create_gpt2(
      { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 },
      {
        tokenEmbedding: value([7, 4]),
        positionEmbedding: value([4, 4]),
        finalNorm: norm(),
        blocks: [{
          attentionNorm: norm(),
          feedForwardNorm: norm(),
          qkv: affine(4, 12),
          attentionOutput: affine(4, 4),
          expand: affine(4, 8),
          contract: affine(8, 4),
        }],
      },
    )
    return model.forward(3)
  })
  return { name: `model_gpt2_${name}_${dtype}`, corpus: 'model', source }
}

function bert(dtype: ModelDType): RegistryEntry {
  const source = withModelSession(dtype, (_session, value) => {
    const affine = (input: number, output: number) => ({
      weight: value([input, output]),
      bias: value([output], 0),
    })
    const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
    const model = create_bert(
      { width: 4, innerWidth: 8, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, typeVocabSize: 2, epsilon: 1e-5 },
      {
        tokenEmbedding: value([7, 4]),
        positionEmbedding: value([4, 4]),
        typeEmbedding: value([2, 4]),
        embeddingNorm: norm(),
        pooler: affine(4, 4),
        blocks: [{
          query: affine(4, 4),
          key: affine(4, 4),
          value: affine(4, 4),
          attentionOutput: affine(4, 4),
          attentionNorm: norm(),
          feedForwardNorm: norm(),
          expand: affine(4, 8),
          contract: affine(8, 4),
        }],
      },
    )
    return model.forward(1, 3)
  })
  return { name: `model_bert_${dtype}`, corpus: 'model', source }
}

function vit(dtype: ModelDType): RegistryEntry {
  const source = withModelSession(dtype, (_session, value) => {
    const affine = (input: number, output: number) => ({
      weight: value([input, output]),
      bias: value([output], 0),
    })
    const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
    const model = create_vit(
      { width: 4, innerWidth: 8, heads: 2, layers: 1, imageSize: 2, patchSize: 1, epsilon: 1e-5 },
      {
        classToken: value([1, 1, 4]),
        positionEmbedding: value([1, 5, 4]),
        patchProjection: { weight: value([4, 3, 1, 1]), bias: value([4], 0) },
        finalNorm: norm(),
        classifier: affine(4, 3),
        blocks: [{
          query: affine(4, 4),
          key: affine(4, 4),
          value: affine(4, 4),
          attentionOutput: affine(4, 4),
          attentionNorm: norm(),
          feedForwardNorm: norm(),
          expand: affine(4, 8),
          contract: affine(8, 4),
        }],
      },
    )
    return model.forward
  })
  return { name: `model_vit_${dtype}`, corpus: 'model', source }
}

function llama(dtype: ModelDType): RegistryEntry[] {
  return withModelSession(dtype, (_session, value) => {
    const model = create_llama(
      { width: 4, innerWidth: 8, heads: 2, kvHeads: 1, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5, ropeTheta: 10_000 },
      {
        tokenEmbedding: value([7, 4]),
        finalNorm: value([4], 1),
        blocks: [{
          attentionNorm: value([4], 1),
          feedForwardNorm: value([4], 1),
          query: value([4, 4]),
          key: value([4, 2]),
          value: value([4, 2]),
          attentionOutput: value([4, 4]),
          gate: value([4, 8]),
          up: value([4, 8]),
          down: value([8, 4]),
        }],
      },
    )
    return [
      { name: `model_llama_prefill_${dtype}`, corpus: 'model', source: model.forward(3) },
      { name: `model_llama_decode_${dtype}`, corpus: 'model', source: model.decode(1) },
    ]
  })
}

export const importedAndModelPrograms: readonly RegistryEntry[] = Object.freeze([
  imported('imported_onnx_generic', 'packages/@affon/onnx/test/fixtures'),
  imported('imported_onnx_spatial', 'packages/@affon/onnx/test/fixtures/spatial'),
  imported('imported_onnx_conv1d', 'packages/@affon/onnx/test/fixtures/conv1d'),
  gpt2('f32', 'prefill'),
  gpt2('bf16', 'prefill'),
  bert('f32'),
  bert('bf16'),
  vit('f32'),
  vit('bf16'),
  ...llama('f32'),
  ...llama('bf16'),
])
