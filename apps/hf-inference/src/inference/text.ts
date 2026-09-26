import type { InferenceModels } from './models.ts'

/** Tokenize, generate greedily, and decode a passage using the loaded text model. */
export function generate_text(
  models: InferenceModels,
  prompt: string,
  max_new_tokens: number,
) {
  const { model, processor } = models.text
  const start = Date.now()
  const ids = processor.encode(prompt)
  if (!ids.length || ids.length > 256)
    throw Error('Use a prompt containing 1–256 tokens.')
  const output = model.generate(ids, max_new_tokens)
  return {
    text: processor.decode(output),
    completion: processor.decode(output.slice(ids.length)),
    prompt_tokens: ids.length,
    generated_tokens: output.length - ids.length,
    elapsed_ms: Date.now() - start,
  }
}
