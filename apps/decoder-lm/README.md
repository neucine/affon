# Decoder LM App

This directory is the canonical decoder-only language-model app for Affon.

Use it when you want to study or test:

- tokenization and packed text windows
- decoder-only model assembly
- causal LM loss and generation
- training, evaluation, checkpointing, and resume
- compiled forward runs
- runtime monitoring and example artifacts

Key files:

- `index.ts`: example-oriented entrypoint
- `training.ts`: training loop and checkpoint logic
- `workflow.ts`: config-driven runner used by the example scripts

Reusable decoder LM model, loss, generation, and token-window primitives live in
`../../packages/lm/src/` rather than this app directory.

Offline graph exports are meant to stay renderer-agnostic. The shared offline
tooling now lives under `../../tools/graph-viewer/`.

Use raw report JSON from `compute.exportReportFile(..., { format: 'json' })`,
then project it into either:

- exact op graph: `projectBundleGraph(bundle, { level: 'op' })`
- collapsed module graph: `projectBundleGraph(bundle, { level: 'module' })`

If you want a quick local page instead of integrating another graph library
immediately, run the offline viewer tool separately and build a self-contained
HTML view:

- `buildBundleGraphViewerHtml(bundle, { level: 'op' | 'module' })`

The workflow step-export path now stops at raw `.json` bundle artifacts.

Module finite diagnostics are off by default. Set `AFFON_NN_DIAGNOSTICS=error`
or run a `*-debug` config through `run.sh` to enable fail-fast finite checks
through `nn.diagnostics`.

WikiText training correctness uses a long-running canary rather than a normal
unit test. Run:

- `AFFON_TRAIN_CONFIG=apps/decoder-lm/configs/train-decoder-lm-wikitext-canary.config.json ./zig-out/bin/affon apps/decoder-lm/train.ts`
- `AFFON_WIKITEXT_CANARY_SUMMARY=apps/decoder-lm/artifacts/wikitext/wikitext-canary-summary.json ./zig-out/bin/affon apps/decoder-lm/check-wikitext-canary.ts`

The checker validates the completed summary history, final validation loss,
minimum loss drop, and epoch-to-epoch validation trend.

The smaller reusable transformer primitives remain under `../../packages/transformers/src/`.
