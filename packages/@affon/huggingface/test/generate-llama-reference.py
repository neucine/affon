"""Regenerate a tiny independent Llama/GQA/RoPE oracle; no downloads required."""
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
root = Path(__file__).parent / 'fixtures' / 'llama'
root.mkdir(exist_ok=True)
torch.manual_seed(1729)
config = LlamaConfig(vocab_size=33, hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32, rope_theta=100000., rms_norm_eps=1e-5, tie_word_embeddings=True, eos_token_id=2, attention_dropout=0.)
config._attn_implementation = 'eager'
model = LlamaForCausalLM(config).eval()
model.save_pretrained(root)
ids = [1, 7, 12, 3, 29]
with torch.no_grad():
    out = model(torch.tensor([ids]), output_hidden_states=True)
    generated = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1,len(ids),dtype=torch.long), max_new_tokens=4, do_sample=False, use_cache=False, pad_token_id=2)[0].tolist()
(root / 'reference.json').write_text(json.dumps(dict(ids=ids, generated=generated, logits=out.logits.tolist(), hidden_states=[v.tolist() for v in out.hidden_states])))
