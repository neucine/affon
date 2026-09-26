import fs from 'std:fs'
import { createHFTokenizerFromJSON } from '../../tokenizers/src/index.ts'

/** Load BERT single/pair templates and right padding using @affon/tokenizers.
 * @param directory Directory containing tokenizer.json and tokenizer_config.json.
 * @returns encode_batch(texts, pairs?) producing IDs, type IDs, and attention masks.
 * @remarks No automatic truncation or general post-processor support.
 */
export function load_bert_processor(directory: string) {
  const spec = JSON.parse(fs.readFileSync(`${directory}/tokenizer.json`))
  const tokenizer = createHFTokenizerFromJSON(spec)
  const config = JSON.parse(fs.readFileSync(`${directory}/tokenizer_config.json`))
  const pad = tokenizer.tokenId(typeof config.pad_token === 'string' ? config.pad_token : config.pad_token?.content)
  if (pad === undefined || (config.padding_side ?? 'right') !== 'right' || spec.post_processor?.type !== 'TemplateProcessing') throw new Error('Expected BERT template processing and right padding')
  function encode_batch(texts: string[], pairs?: string[]) {
    if (!texts.length || (pairs && pairs.length !== texts.length)) throw new Error('Invalid BERT text batch')
    const template = pairs ? spec.post_processor.pair : spec.post_processor.single
    if (!Array.isArray(template)) throw new Error('Missing BERT template')
    const rows = texts.map((text, index) => {
      const sequences: Record<string, number[]> = { A: tokenizer.encode(text) }
      if (pairs) sequences.B = tokenizer.encode(pairs[index])
      const ids: number[] = [], types: number[] = []
      for (const part of template) {
        const entry = part.Sequence ?? part.SpecialToken
        if (!entry || !Number.isInteger(entry.type_id) || entry.type_id < 0) throw new Error('Invalid BERT template entry')
        const values: number[] = part.Sequence ? sequences[entry.id] : spec.post_processor.special_tokens[entry.id]?.ids
        if (!values || values.some(id => tokenizer.token(id) === undefined)) throw new Error('Invalid BERT template token')
        ids.push(...values); types.push(...values.map(() => entry.type_id))
      }
      return { ids, types }
    })
    const length = Math.max(...rows.map(row => row.ids.length))
    return {
      input_ids: rows.map(row => [...row.ids, ...Array(length - row.ids.length).fill(pad)]),
      token_type_ids: rows.map(row => [...row.types, ...Array(length - row.types.length).fill(0)]),
      attention_mask: rows.map(row => [...row.ids.map(() => 1), ...Array(length - row.ids.length).fill(0)]),
    }
  }
  return { encode_batch }
}
