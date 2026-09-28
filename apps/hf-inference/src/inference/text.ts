import type { InferenceModels } from './models.ts'

/** Tokenize, generate greedily, and decode a passage using the loaded text model. */
function prepare_text(
  models: InferenceModels,
  prompt: string,
  max_new_tokens: number,
  model_key?: string,
) {
  const key = model_key ?? models.default_text_model ?? 'distilgpt2'
  const selected = Object.hasOwn(models.texts, key) ? models.texts[key] : undefined
  if (!selected) throw Error('Unknown or unavailable text model')
  const { model, processor, chat, id: modelId } = selected
  if (!Number.isInteger(max_new_tokens) || max_new_tokens < 1 || max_new_tokens > 64) throw Error('Use 1–64 new tokens.')
  if (chat && !('encode_chat' in processor)) throw Error('Chat model requires a chat processor')
  const start = Date.now()
  const ids = chat && 'encode_chat' in processor
    ? processor.encode_chat([{role: 'user', content: prompt}])
    : processor.encode(prompt)
  if (!ids.length || ids.length > 256)
    throw Error('Use a prompt containing 1–256 tokens including chat formatting.')
  function result(output: number[]) {
    const completion = processor.decode(output.slice(ids.length), {skipSpecialTokens: chat})
    return {
      model: modelId,
      text: chat ? completion : processor.decode(output),
      completion,
      truncated: output.length - ids.length === max_new_tokens && output[output.length - 1] !== model.config.eos_token_id,
      prompt_tokens: ids.length,
      generated_tokens: output.length - ids.length,
      elapsed_ms: Date.now() - start,
    }
  }
  return {model, ids, result}
}

/** Preserve the buffered API for existing callers. */
export function generate_text(models: InferenceModels, prompt: string, budget: number, key?: string) {
  const {model, ids, result} = prepare_text(models, prompt, budget, key)
  return result(model.generate(ids, budget))
}

/** One cached decode step per call. Decode the complete prefix to preserve UTF-8
 * token boundaries; callers replace their displayed text with each snapshot. */
export function create_text_generation(models: InferenceModels, prompt: string, budget: number, key?: string) {
  const {model, ids, result} = prepare_text(models, prompt, budget, key)
  const session = model.create_session(), output = [...ids]
  let done = false
  const close = () => { done = true; session.reset() }
  return {
    close,
    next() {
      if (done) throw Error('Generation has finished')
      try {
        const input = output.length === ids.length ? ids : [output[output.length - 1]]
        const logits = session.forward(input).logits
        const row = (logits.slice([0, input.length - 1, ':']).to_array() as number[]).flat(Infinity) as number[]
        let best = 0
        for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i
        output.push(best)
        done = best === model.config.eos_token_id || output.length - ids.length === budget
        const snapshot = result(output)
        // A byte-level tokenizer may end mid-codepoint before the next token.
        if (!done) snapshot.text = snapshot.text.replace(/\uFFFD+$/, '')
        if (done) session.reset()
        return {...snapshot, done}
      } catch (error) { close(); throw error }
    },
  }
}
