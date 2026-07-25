# @affon/tokenizers

Tokenizer compatibility package.

This package owns tokenizer implementations and ecosystem adapters. `affon:dataset`
depends only on the structural `TextTokenizer` contract for encoding, padding,
batching, and tensorization.

Owned here:

- lookup tokenizers for small local vocabularies
- Hugging Face tokenizer compatibility
- SentencePiece compatibility for simplified JSON models
- BPE, WordPiece, Unigram, and byte-level edge cases
- tokenizer import/export helpers
- compatibility fixtures and parity tests

Not owned here:

- dataset pipeline use of tokenizers, which belongs in `affon:dataset`
- text batching, padding, and tensorization, which belong in `affon:dataset`
- LM-specific recipes such as prompt templates, chat templates, and corpus packing, which belong in `../lm/` or `../../../apps/`

Use the package entrypoint for tokenizer compatibility APIs:

```ts
import { createHFTokenizerFromFile, createLookupTokenizer } from '@affon/tokenizers'
```
