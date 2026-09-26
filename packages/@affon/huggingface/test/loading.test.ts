import { describe, expect, test } from 'std:test'
import { load_model } from '../src/index.ts'
import type { ModelTask } from '../src/index.ts'

const root = 'packages/@affon/huggingface/test/fixtures'
describe('HF task/model dispatch', () => {
  for (const [family, task] of [
    ['bert', 'text-generation'], ['gpt2', 'feature-extraction'],
    ['vit', 'feature-extraction'], ['unknown', 'image-classification'],
  ] as [string, ModelTask][]) {
    test(`rejects ${family}/${task} before looking for weights`, () => {
      expect(() => load_model(`${root}/${family}`, { task })).toThrow(`Unsupported HF model/task: ${family}/${task}`)
    })
  }
})
