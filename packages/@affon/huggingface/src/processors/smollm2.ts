import fs from 'std:fs'
import { createHFTokenizerFromFile } from '../../../tokenizers/src/index.ts'

export const SMOLLM2_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
export interface SmolLM2Message { role: 'system' | 'user' | 'assistant'; content: string }

/** The pinned SmolLM2 template, implemented without executing arbitrary Jinja. */
export function format_smollm2_chat(messages: readonly SmolLM2Message[], add_generation_prompt = true) {
  if (!messages.length || messages.some(m => !['system', 'user', 'assistant'].includes(m.role) || typeof m.content !== 'string')) throw Error('Expected nonempty SmolLM2 chat messages')
  let text = messages[0].role === 'system' ? '' : '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n'
  for (const message of messages) text += `<|im_start|>${message.role}\n${message.content}<|im_end|>\n`
  return text + (add_generation_prompt ? '<|im_start|>assistant\n' : '')
}

export function load_smollm2_processor(directory: string) {
  const config = JSON.parse(fs.readFileSync(`${directory}/tokenizer_config.json`))
  if (config.chat_template !== SMOLLM2_CHAT_TEMPLATE || config.bos_token !== '<|im_start|>' || config.eos_token !== '<|im_end|>') throw Error('Unsupported SmolLM2 chat template or special tokens')
  const tokenizer = createHFTokenizerFromFile(`${directory}/tokenizer.json`, { specialTokens: { bos: '<|im_start|>', eos: '<|im_end|>', pad: '<|im_end|>', unk: '<|endoftext|>' } })
  if (tokenizer.tokenId('<|im_start|>') !== 1 || tokenizer.tokenId('<|im_end|>') !== 2) throw Error('Unexpected SmolLM2 chat token IDs')
  return { ...tokenizer, encode_chat: (messages: readonly SmolLM2Message[]) => tokenizer.encode(format_smollm2_chat(messages)) }
}
