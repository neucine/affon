# @affon/transformers

Transformer architecture package for Affon.

This package owns reusable transformer-family architecture blocks. It should stay focused on structures that can be reused by language, vision, multimodal, and other transformer-shaped packages.

## Package Boundary

Owned here:

- attention blocks
- decoder and encoder-style block pieces
- feed-forward blocks
- transformer input embedding pieces
- transformer mask and position helpers

Not owned here:

- LM training workflows
- generation policies and sampling recipes
- tokenizer ecosystem compatibility
- corpus loading, token caching, and LM token-window packing
- domain apps such as image classification or decoder-LM training
- runtime tensor/autograd/device primitives

Use the package entrypoint for architecture APIs:

```ts
import { DecoderBlock, SelfAttention, causal_mask } from '@affon/transformers'
```

## Reusable Architecture APIs

Current reusable pieces in `src/`:

- `src/decoder/attention.ts`
- `src/decoder/block.ts`
- `src/decoder/embedding.ts`
- `src/decoder/feedforward.ts`
- `src/sequence.ts`

Text preprocessing should prefer `affon:dataset` directly. Tokenizer compatibility belongs in `../tokenizers/`; reusable LM corpus packing belongs in `../lm/`; app-specific workflow code belongs in `../../apps/`.

Numerical assertion policy is owned by `affon:nn` diagnostics. Transformer blocks expose diagnostic assertion sites, but do not own enable/disable policy or diagnostic formatting.

## Decoder-LM App

The decoder-LM workload now lives under `../../apps/decoder-lm/`. It is intentionally app code, not the stable `@affon/transformers` package API.

It exists to exercise Affon end to end across:

- compute semantics
- runtime execution
- autograd
- tensor kernels
- compilation
- checkpointing and resume
- memory behavior
- module/package loading
- observability and diagnostics

Keep app workflow code out of the transformer package entrypoint. Reusable language-model pieces belong in `../lm/`; reusable transformer architecture pieces belong here.

- `../../apps/decoder-lm/src/index.ts`
- `../../apps/decoder-lm/src/training.ts`
- `../../apps/decoder-lm/src/workflow.ts`
- `../../apps/decoder-lm/train.ts`

Loss usage:

```ts
import { CausalLMLoss } from '../lm/src/causal-lm.ts'
import { DecoderModel } from '../lm/src/model.ts'

const model = DecoderModel(vocabSize, dModel, {
  numLayers: 2,
  numHeads: 4,
})
const criterion = CausalLMLoss()
const inputs = tokenIds.slice([':', '0:-1'])
const loss = criterion(model(inputs), tokenIds)
```

Training helpers:

```ts
import dataset from 'affon:dataset'
import { DecoderModel } from '../lm/src/model.ts'
import { trainDecoderLM } from '../../apps/decoder-lm/src/index.ts'

const windows = dataset.text.encoded(tokenRows).window({
  seqLen: 128,
  stride: 64,
  joinWithTokenId: eosId,
}).toArray()

const model = DecoderModel(vocabSize, 128, {
  numLayers: 4,
  numHeads: 4,
  hiddenDim: 512,
  dropout: 0.1,
})

const result = trainDecoderLM(model, windows, {
  seqLen: 128,
  batchSize: 8,
  epochs: 3,
  lr: 3e-4,
  checkpointEveryEpochs: 1,
})
```

For faster debugging on large corpora, training can cap the number of train and eval batches per epoch:

```ts
const result = trainDecoderLM(model, windows, {
  seqLen: 128,
  batchSize: 8,
  epochs: 1,
  lr: 1e-4,
  maxTrainBatchesPerEpoch: 50,
  maxEvalBatches: 5,
})
```

Checkpoint helpers:

```ts
import { loadDecoderLMCheckpoint, saveDecoderLMCheckpoint } from '../../apps/decoder-lm/src/index.ts'

saveDecoderLMCheckpoint('checkpoints/run-1-epoch-1', result.checkpoints[0], {
  metadata: { tokenizer: 'lookup' },
})

const loaded = loadDecoderLMCheckpoint('checkpoints/run-1-epoch-1', model)
```

Config-driven training:

```sh
bash apps/decoder-lm/run.sh helloworld
```

The shell runner uses the decoder-LM example module directly, not the package entrypoint.

Named runner:

```sh
bash apps/decoder-lm/run.sh helloworld
bash apps/decoder-lm/run.sh tinystories
bash apps/decoder-lm/run.sh wikitext
bash apps/decoder-lm/run.sh wikitext-debug
```

Progress reporting:

```json
{
  "report": {
    "progress": {
      "trainPhase": false,
      "trainLossEveryBatches": 10,
      "evalEveryBatches": 10,
      "epochSummary": true
    }
  }
}
```

Example defaults:
- `helloworld` prints train phases and every batch loss
- `tinystories` prints train phases and every batch loss
- `wikitext` prints every 10 training batches, every 10 eval batches, and epoch summaries

Additional workflow templates:
- [train-decoder-lm-helloworld.config.json](../../apps/decoder-lm/configs/train-decoder-lm-helloworld.config.json)
- [train-decoder-lm-tinystories.config.json](../../apps/decoder-lm/configs/train-decoder-lm-tinystories.config.json)
- [train-decoder-lm-wikitext.config.json](../../apps/decoder-lm/configs/train-decoder-lm-wikitext.config.json)

The TinyStories template now points at a bundled real sample under:
- [data/tinystories-tokenizer.json](../../apps/decoder-lm/data/tinystories-tokenizer.json)
- [data/tinystories-train.txt](../../apps/decoder-lm/data/tinystories-train.txt)
- [data/tinystories-validation.txt](../../apps/decoder-lm/data/tinystories-validation.txt)

The WikiText template now points at bundled `WikiText-2` sample files under:
- [data/wikitext-gpt2-tokenizer.json](../../apps/decoder-lm/data/wikitext-gpt2-tokenizer.json)
- [data/wikitext-2-train.txt](../../apps/decoder-lm/data/wikitext-2-train.txt)
- [data/wikitext-2-validation.txt](../../apps/decoder-lm/data/wikitext-2-validation.txt)

The full `wikitext` config is the quality-oriented preset:
- 8 training epochs
- `dModel: 256`, `numLayers: 6`, `numHeads: 8`, `hiddenDim: 1024`
- `seqLen: 256`, `stride: 128`
- Metal by default
- literal WikiText placeholders such as `<unk>`, `@-@`, and `@,@` are cleaned before tokenization
- keyed token caches keep raw and cleaned corpora separate

Use `wikitext-debug` only for shorter smoke/resume loops.

The workflow can also:
- emit periodic sample generations during training
- write a JSON run summary with loss/perplexity history
- save checkpoint manifests and weights
- resume model training from a saved checkpoint prefix

Resume example:

```json
{
  "checkpoint": {
    "prefix": "apps/decoder-lm/artifacts/tinystories/run",
    "everyNEpochs": 2,
    "resumeFrom": "apps/decoder-lm/artifacts/tinystories/run-epoch-10"
  }
}
```

This restores model weights, optimizer state, and continues epoch/step numbering from that checkpoint.

Model config can also enable decoder dropout with a single shared `dropout` value.
It is applied to embeddings plus the attention and feed-forward residual branches:

```json
{
  "model": {
    "dModel": 256,
    "numLayers": 8,
    "numHeads": 8,
    "hiddenDim": 1024,
    "dropout": 0.1
  }
}
```

The workflow also supports a combined warmup-plus-cosine schedule:

```json
{
  "training": {
    "lrSchedule": {
      "kind": "warmup_cosine",
      "start": 0.000001,
      "peak": 0.00001,
      "end": 0.000002,
      "warmup": { "unit": "step", "value": 500 },
      "total": { "unit": "epoch", "value": 2 }
    }
  }
}
```

Training can also select `adamw` explicitly when weight decay is part of the recipe:

```json
{
  "training": {
    "lr": 0.00001,
    "optimizer": {
      "kind": "adamw",
      "weightDecay": 0.01
    }
  }
}
```

To increase effective batch size without changing memory use, workflow training also supports
gradient accumulation:

```json
{
  "training": {
    "batchSize": 4,
    "gradientAccumulationSteps": 4
  }
}
```

To resume from the latest best completed epoch instead of an exact checkpoint, use:

```json
{
  "checkpoint": {
    "prefix": "apps/decoder-lm/artifacts/wikitext/run",
    "resumeStrategy": "best_epoch",
    "everyNEpochs": 1
  }
}
```

This reads the workflow summary, picks the latest epoch with the best loss
(`valLoss` when available, otherwise `trainLoss`), loads
`<prefix>-epoch-<bestEpoch>`, and starts training again from the next epoch.

Training config can also provide an `lrSchedule` instead of a fixed `lr`. Supported schedule kinds are:
- `constant`
- `warmup_constant`
- `linear`
- `cosine`
- `step`

Example:

```json
{
  "training": {
    "epochs": 4,
    "batchSize": 8,
    "lrSchedule": {
      "kind": "step",
      "base": 0.0001,
      "gamma": 0.5,
      "every": { "unit": "epoch", "value": 1 }
    }
  }
}
```

Warmup example:

```json
{
  "training": {
    "epochs": 2,
    "batchSize": 8,
    "lrSchedule": {
      "kind": "warmup_constant",
      "start": 0.000001,
      "lr": 0.00003,
      "duration": { "unit": "step", "value": 500 }
    }
  }
}
```

The Hello World example config uses a small Hugging Face `tokenizer.json` artifact:
- [data/helloworld-tokenizer.json](../../apps/decoder-lm/data/helloworld-tokenizer.json)

It is still a small Hello World demo vocab, just packaged in the same artifact format the workflow expects. For real corpora, the preferred path is to point config at an existing external tokenizer artifact rather than maintain a manual lookup vocab in config.

Example:

```sh
bash apps/decoder-lm/run.sh helloworld

Use `wikitext-debug` for shorter Metal/resume feedback loops while keeping the same model shape and resume path as the full `wikitext` example:

```bash
bash apps/decoder-lm/run.sh wikitext-debug
```
```

The example runner enables native stack printing by default:
- it sets `AFFON_NATIVE_STACK_TRACE=1` unless you already provided a value
- native-origin failures during the example run will print the attached Zig native stack
- normal examples leave module finite diagnostics off; set `AFFON_NN_DIAGNOSTICS=error` to scan matching checked tensors, or use `wikitext-debug`

LLM-specific training and generation workflows should live in a separate `@affon/llm` package.
