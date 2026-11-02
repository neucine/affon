# Local Apple M4 inference baseline

Build: ReleaseFast. One warmup, 5 measured runs per process. Batch one; f32.

| Workload / backend | Median inference | Model construction | Runtime setup | OS peak RSS | Live tensor range after warmup |
| --- | ---: | ---: | ---: | ---: | ---: |
| [gpt2-metal](gpt2-metal.json) | 0.435 s | 293 ms | 376 ms | 945.0 MiB | 316.0–316.0 MiB |

GPT-2: five prompt tokens, 16 new greedy tokens, explicit full-prefix Programs with one-token output windows. ViT (when measured): 320×256 synthetic RGB input; forward includes logit readback and excludes preprocessing.

OS peak RSS covers the entire process, including startup/loading. Construction excludes snapshot validation; neither is guaranteed cold-disk timing. Memory ranges are six samples for the default five-run benchmark, not a leak-freedom claim.

See [methodology](../../README.md) and [host/run metadata](run.json). Raw reports include preprocessing, warmup, all samples, and memory observations.
