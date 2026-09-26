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

HF ByteLevel BPE preserves whitespace and uses GPT-2 token boundaries before
applying merge ranks. `add_prefix_space` and `use_regex` are respected. Added
tokens are matched before pre-tokenization, with longest matches preferred and
`single_word`, `lstrip`, and `rstrip` constraints applied. Added IDs are included
in vocabulary lookup/size; only special tokens are removed by `skipSpecialTokens`.

BERT tokenization supports `BertNormalizer` (text cleaning, Chinese-character
boundaries, optional lowercasing and accent stripping) and `BertPreTokenizer`
punctuation splitting. `encode` continues to return raw token IDs; it does not
automatically apply serialized postprocessor templates. The HF audit app owns
the current BERT template/padding adapter, including paired inputs.

This remains partial HF compatibility: arbitrary normalizer/postprocessor chains,
normalized added-token matching against transformed text, and arbitrary decoder
chains are not implemented. Unsupported normalizer types are rejected. Do not
infer compatibility from the model type alone.

The offline ByteLevel fixture is generated from synthetic vocabulary/merges by
`test/generate-hf-bytelevel-reference.py` with `tokenizers==0.22.2`. It covers
Unicode, whitespace, contractions, added-token boundaries, and ByteLevel options.
Run from the repository root:

```sh
./zig-out/bin/affon test packages/@affon/tokenizers/test
```
