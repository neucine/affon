"""Measure explicit full-prefix GPT-2 Program execution on Metal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--affon', required=True)
parser.add_argument('--model-dir', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
identifier = uuid.uuid4().hex
runner = repo / f'apps/hf-inference/src/benchmark/profile-prefix-{identifier}.ts'
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
try:
    runner.write_text("""import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_gpt2 } from '../../../../packages/@affon/huggingface/src/index.ts'
import { CausalProgramRuntime } from '../inference/program-runtime.ts'
const model = load_gpt2(getEnv('PROFILE_MODEL')!)
const runtime = new CausalProgramRuntime(model, 'metal')
const ids = Array.from({length: 128}, () => 464)
let result = runtime.forward(ids, ids.length - 1)
result.logits.to_array(); result.logits.dispose(); for (const value of result.hidden_states) value.dispose()
const start = Date.now()
for (let index = 0; index < 32; index++) {
  ids.push(464)
  result = runtime.forward(ids, ids.length - 1)
  result.logits.to_array(); result.logits.dispose(); for (const value of result.hidden_states) value.dispose()
}
runtime.dispose()
fs.writeFileSync(getEnv('PROFILE_OUTPUT')!, JSON.stringify({ elapsed_ms: Date.now() - start, steps: 32 }, null, 2))
""")
    subprocess.run([str(Path(args.affon).resolve()), str(runner.relative_to(repo))], cwd=repo, check=True,
                   env={**os.environ, 'PROFILE_MODEL': str(Path(args.model_dir).resolve()), 'PROFILE_OUTPUT': str(output)})
    report = json.loads(output.read_text())
    source = repo / 'packages/@affon/models/src/gpt2/model.ts'
    report.update({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'binary_sha256': hashlib.sha256(Path(args.affon).read_bytes()).hexdigest(),
                   'method': 'One 128-token warmup, then 32 forced-token full-prefix Programs with one-token output windows and synchronized logit readback. No KV cache is claimed.'})
    output.write_text(json.dumps(report, indent=2) + '\n')
finally:
    runner.unlink(missing_ok=True)
