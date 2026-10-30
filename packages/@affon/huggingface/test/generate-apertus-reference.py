"""Regenerate a small Apertus fixture with independent Hugging Face Transformers.

Run with torch, transformers, and safetensors installed:
  python packages/@affon/huggingface/test/generate-apertus-reference.py
"""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import ApertusConfig, ApertusForCausalLM

root = Path(__file__).parent / "fixtures" / "apertus"
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(2217)
config = ApertusConfig(
    vocab_size=32, hidden_size=16, intermediate_size=24,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=64, rope_theta=12000000, rms_norm_eps=1e-5,
    hidden_act="xielu", qk_norm=True, post_norm=False,
    attention_bias=False, mlp_bias=False, tie_word_embeddings=False,
    bos_token_id=1, eos_token_id=2, pad_token_id=3,
    rope_scaling={"rope_type": "llama3", "factor": 8.0,
                  "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8},
)
model = ApertusForCausalLM(config).eval()
ids = torch.tensor([[1, 7, 9, 4]])
with torch.no_grad():
    output = model(ids, output_hidden_states=True, use_cache=False)
(root / "config.json").write_text(config.to_json_string())
save_file({name: value.contiguous() for name, value in model.state_dict().items()}, root / "model.safetensors")
bf16 = root.parent / "apertus-bf16"
bf16.mkdir(exist_ok=True)
(bf16 / "config.json").write_text(config.to_json_string())
save_file({name: value.to(torch.bfloat16).contiguous() for name, value in model.state_dict().items()}, bf16 / "model.safetensors")
with torch.no_grad():
    bf16_output = model.to(torch.bfloat16)(ids, output_hidden_states=True, use_cache=False)
(bf16 / "reference.json").write_text(json.dumps({
    "ids": ids[0].tolist(), "logits": bf16_output.logits.float().tolist(),
    "hidden_states": [value.float().tolist() for value in bf16_output.hidden_states],
}, indent=2) + "\n")
(root / "reference.json").write_text(json.dumps({
    "ids": ids[0].tolist(),
    "logits": output.logits.tolist(),
    "hidden_states": [value.tolist() for value in output.hidden_states],
    "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
}, indent=2) + "\n")
