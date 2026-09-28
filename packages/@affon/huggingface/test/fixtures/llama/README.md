# Tiny Llama oracle

Synthetic random weights, seed 1729, generated with PyTorch 2.9.0 and
Transformers 4.56.2 using `../../generate-llama-reference.py`. No pretrained
weights or downloaded data are included.

Two layers, width 16, four query heads, two K/V heads, head dimension four,
SiLU-gated MLP width 24, tied embeddings, full unscaled RoPE theta 100000.
`reference.json` stores full logits, every hidden state, and uncached greedy IDs
from CPU f32 eager attention. Tests use 1e-5 absolute tolerance and exact token IDs.
