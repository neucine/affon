"""Warm HF/PyTorch MPS baseline for the bounded Affon Whisper benchmark."""
import argparse
import hashlib
import io
import json
import platform
import time
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--wav', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
args = parser.parse_args()
assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback benchmark'
torch.set_num_threads(4)
policy = json.loads((args.model / 'whisper.json').read_text())
model = WhisperForConditionalGeneration.from_pretrained(
    args.model / 'source', local_files_only=True,
    attn_implementation=args.attention, dtype=torch.float32,
).eval().to('mps')
processor = WhisperProcessor.from_pretrained(args.model / 'source', local_files_only=True)
raw = args.wav.read_bytes()  # File I/O and loading are outside request timing.


def sync_time():
    torch.mps.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def run():
    start = sync_time()
    with wave.open(io.BytesIO(raw)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32) / 32768
    # HF's default CPU feature extraction, followed by transfer to MPS.
    features = processor.feature_extractor(samples, sampling_rate=16000, return_tensors='pt').input_features.to('mps')
    prepared = sync_time()
    encoded = model.model.encoder(features, return_dict=True)
    encoded_at = sync_time()
    ids = model.generate(
        input_features=features, encoder_outputs=encoded,
        decoder_input_ids=torch.tensor([policy['prefix']], device='mps'),
        max_new_tokens=policy['width'] - len(policy['prefix']),
        do_sample=False, num_beams=1, use_cache=True, return_timestamps=False,
        return_dict_in_generate=True,
        suppress_tokens=policy['suppress_tokens'],
        begin_suppress_tokens=policy['begin_suppress_tokens'],
    ).sequences[0].cpu().tolist()
    text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    end = sync_time()
    return dict(tokens=ids, text=text, truncated=ids[-1] != policy['eos'],
                preprocessing_ms=(prepared-start)*1000, encoder_ms=(encoded_at-prepared)*1000,
                decoder_ms=(end-encoded_at)*1000, inference_ms=(end-prepared)*1000,
                elapsed_ms=(end-start)*1000)


warmup = run()
runs = [run() for _ in range(3)]
result = dict(torch=torch.__version__, transformers=transformers.__version__,
              platform=platform.platform(), device='mps', dtype='float32',
              attention=args.attention, cpu_threads=torch.get_num_threads(),
              model=policy['model'], revision=policy['revision'], wav=str(args.wav),
              wav_sha256=hashlib.sha256(raw).hexdigest(), warmup=1,
              warmup_tokens=warmup['tokens'], runs=runs)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(runs, indent=2))
