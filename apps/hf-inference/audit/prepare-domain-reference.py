"""Prepare pinned BERT embedding or ViT classification reference artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoTokenizer, BertModel, ViTForImageClassification, ViTImageProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['bert', 'vit'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('A full lowercase commit SHA is required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    options = dict(revision=args.revision, trust_remote_code=False)
    model_class = BertModel if args.family == 'bert' else ViTForImageClassification
    model = model_class.from_pretrained(args.model, dtype=torch.float32, attn_implementation='eager', **options).cpu().eval()
    model.save_pretrained(output, safe_serialization=True)
    tensors, cases = {}, []
    if args.family == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
        tokenizer.save_pretrained(output)
        for index, texts in enumerate([['Hello world!'], ['CAFÉ, hello?', 'A short sentence.'], ['A much longer sentence for padding checks.', 'Hi!']]):
            encoded = tokenizer(texts, padding=True, return_tensors='pt')
            with torch.inference_mode():
                result = model(**encoded, output_hidden_states=True)
                mask = encoded['attention_mask'].unsqueeze(-1)
                pooled = (result.last_hidden_state * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            tensors[f'case_{index}.output'] = result.last_hidden_state.contiguous()
            tensors[f'case_{index}.pooled'] = pooled.contiguous()
            tensors[f'case_{index}.pooler'] = result.pooler_output.contiguous()
            for layer, hidden in enumerate(result.hidden_states):
                tensors[f'case_{index}.hidden_{layer}'] = hidden.contiguous()
            cases.append({'texts': texts, **{name: value.tolist() for name, value in encoded.items()}})
    else:
        processor = ViTImageProcessor.from_pretrained(args.model, **options)
        processor.save_pretrained(output)
        for index, (height, width) in enumerate([(224, 224), (260, 320)]):
            # Deterministic RGB input, no downloaded media; same input recipe is
            # independently implemented in the native audit.
            y, x, c = np.indices((height, width, 3))
            pixels = ((x * 3 + y * 5 + c * 47) % 256).astype(np.uint8)
            encoded = processor(images=Image.fromarray(pixels), return_tensors='pt')
            with torch.inference_mode():
                result = model(**encoded, output_hidden_states=True)
            tensors[f'case_{index}.pixels'] = encoded['pixel_values'].contiguous()
            tensors[f'case_{index}.output'] = result.logits.contiguous()
            for layer, hidden in enumerate(result.hidden_states):
                tensors[f'case_{index}.hidden_{layer}'] = hidden.contiguous()
            cases.append({'height': height, 'width': width, 'top1': result.logits.argmax(-1).tolist()})
    save_file({name: value.clone() for name, value in tensors.items()}, str(output / 'reference.safetensors'))
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != 'reference.json' and not path.name.startswith('audit-'):
            with path.open('rb') as handle:
                hashes[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {
        'format': 'affon-hf-domain-reference/v1', 'family': args.family,
        'model_id': args.model, 'revision': args.revision, 'dtype': 'float32',
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers', 'safetensors', 'pillow']},
        'artifact_note': 'HF model reserialized as f32 SafeTensors; no Affon-specific tensor-name conversion.',
        'sha256': hashes, 'cases': cases,
    }
    (output / 'reference.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(cases)} {args.family} cases to {output}')


if __name__ == '__main__':
    main()
