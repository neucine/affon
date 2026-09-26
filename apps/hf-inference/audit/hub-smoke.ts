import { getEnv } from 'std:process'
import { from_pretrained } from '../../../packages/@affon/huggingface/src/index.ts'

const options = { cache_dir: getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache', local_files_only: getEnv('AFFON_HF_OFFLINE') === '1' }
if (getEnv('AFFON_HF_FAMILY') === 'vit') {
  const { model, processor, directory } = await from_pretrained('google/vit-base-patch16-224', {
    ...options, revision: '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3', task: 'image-classification',
  })
  const rgb = Array.from({ length: 256 }, (_, y) => Array.from({ length: 320 }, (_, x) => Array.from({ length: 3 }, (_, c) => (x * 3 + y * 5 + c * 47) % 256)))
  const logits = (model.forward(processor.process(rgb)).output.to_array() as number[][])[0]
  if (logits.some(x => !Number.isFinite(x))) throw new Error('Nonfinite ViT logits')
  const top1 = logits.indexOf(Math.max(...logits))
  console.log(JSON.stringify({ directory, task: 'image-classification', top1, label: model.config.id2label[top1], logits: logits.length }))
} else {
  const { model, processor, directory } = await from_pretrained('distilbert/distilgpt2', {
    ...options, revision: '2290a62682d06624634c1f46a6ad5be0f47f38aa', task: 'text-generation',
  })
  const ids = processor.encode('The future of computing is')
  const output = model.generate(ids, 4)
  console.log(JSON.stringify({ directory, task: 'text-generation', ids: output, text: processor.decode(output) }))
}
