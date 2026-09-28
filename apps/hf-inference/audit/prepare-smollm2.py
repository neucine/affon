"""Prepare independent f32 CPU/eager references from the pinned BF16 checkpoint.
Weights remain original BF16. Python is only an audit dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

SPECS = {
    '135M': '12fd25f77366fa6b3b4b768ec3050bf629380bac',
    '360M': 'a10cc1512eabd3dde888204e902eca88bddb4951',
    '1.7B': '31b70e2e869a7173562077fd711b654946d38674',
}
p = argparse.ArgumentParser()
p.add_argument('--size', choices=SPECS, default='135M')
p.add_argument('--directory', required=True, help='Local pinned snapshot (original weights/tokenizer/config)')
args = p.parse_args()
MODEL, REVISION = f'HuggingFaceTB/SmolLM2-{args.size}-Instruct', SPECS[args.size]
root = Path(args.directory)
def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
snapshot = json.loads((root / 'affon-snapshot.json').read_text())
if snapshot['model_id'] != MODEL or snapshot['revision'] != REVISION:
    raise ValueError('Expected the pinned SmolLM2 Hub snapshot')
for entry in snapshot['files']:
    if Path(entry['name']).name != entry['name']:
        raise ValueError('Unexpected snapshot artifact')
    if sha256(root / entry['name']) != entry['sha256']:
        raise ValueError('Snapshot checksum mismatch: ' + entry['name'])
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
prompts = ['What is the capital of France?', 'Rewrite politely: Give me the report now.', 'What is 123 + 456? café 中文 １２３\n\tThanks!', 'Say hello.']
refs, cases = {}, []
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        messages = [{'role': 'user', 'content': prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        x = torch.tensor([ids])
        result = model(x, output_hidden_states=True, use_cache=False)
        refs[f'case.{i}.logits'] = result.logits.contiguous()
        # All layers make normalization/RoPE/layout errors observable independently of argmax.
        for j, hidden in enumerate(result.hidden_states):
            refs[f'case.{i}.hidden.{j}'] = hidden.contiguous()
        generated = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=2)[0].tolist()
        cases.append(dict(prompt=prompt, formatted=formatted, ids=ids, generated=generated, completion=tokenizer.decode(generated[len(ids):], skip_special_tokens=True), hidden_count=len(result.hidden_states)))
        print(i, cases[-1]['completion'], flush=True)
save_file(refs, str(root / 'smollm2-reference.safetensors'))
manifest = dict(model=MODEL, revision=REVISION, torch=torch.__version__, transformers=transformers.__version__, dtype='float32', attention='eager', cases=cases, sha256={name: sha256(root / name) for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']})
(root / 'smollm2-reference.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
