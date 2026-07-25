# @affon/lm

Language-model family package.

This package owns reusable LM-family APIs that are broader than one app but more specific than transformer architecture blocks.

Owned here:

- decoder-only LM assembly over `@affon/transformers` blocks
- causal LM loss and eval-loss helpers
- generation loops and sampling options
- token-window packing and batching primitives
- packed text-corpus helpers that produce LM token windows

Packed corpus configs can apply small literal preprocessing before tokenization:

```json
{
  "preprocess": {
    "replacements": [{ "from": "<unk>", "to": "" }],
    "normalizeWhitespace": true
  }
}
```

Token-row caches may also carry an explicit cache key. When a config supplies
`tokenCacheKey` or `validationTokenCacheKey`, older caches with a different key
are ignored and regenerated from text.

Not owned here:

- generic attention, embedding, and decoder blocks, which belong in `../transformers/`
- complete runnable training workloads, which should move toward `../../../apps/`
- dataset source adapters and tokenizer compatibility layers, which should stay in `affon:dataset`, `../tokenizers/`, or apps
- runtime tensor/autograd primitives, which belong in `affon`

Primary ML-facing helpers should use `snake_case`; this package does not keep camelCase compatibility aliases.
