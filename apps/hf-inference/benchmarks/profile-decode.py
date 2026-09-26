"""Attribute eager GPT-2 decode wall time to compute API calls (Metal calls currently synchronize)."""
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
source = repo / 'packages/@affon/huggingface/src/gpt2.ts'
text = source.read_text()
start = text.index('import {\n')
end = text.index("} from 'affon:compute'", start) + len("} from 'affon:compute'")
names = text[start:end].split('{')[1].split('}')[0].replace('\n', '').replace(' ', '').strip(',').split(',')
wrappers = "import { " + ', '.join(n + ' as native_' + n for n in names) + " } from 'affon:compute'\n"
wrappers += 'export const profile: Record<string, { calls: number; ms: number }> = {}\n'
for name in names:
    wrappers += f"const {name}: typeof native_{name} = ((...args: any[]) => {{ const t = Date.now(); try {{ return (native_{name} as any)(...args) }} finally {{ const s = profile['{name}'] ??= {{ calls: 0, ms: 0 }}; s.calls++; s.ms += Date.now() - t }} }}) as any\n"
identifier = uuid.uuid4().hex
instrumented = source.with_name(f'gpt2-profile-{identifier}.ts')
runner = repo / f'apps/hf-inference/src/benchmark/profile-{identifier}.ts'
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
try:
    instrumented.write_text(text[:start] + wrappers + text[end:])
    runner.write_text("""import fs from 'std:fs'
import { getEnv } from 'std:process'
import { load_gpt2, profile } from '../../../../packages/@affon/huggingface/src/""" + instrumented.name + """'
const model = load_gpt2(getEnv('PROFILE_MODEL')!, 'metal')
const session = model.create_session()
session.forward(Array.from({length:128}, () => 464)).logits.to_array()
for (const key of Object.keys(profile)) delete profile[key]
const start = Date.now()
for (let i=0;i<32;i++) session.forward([464]).logits.to_array()
fs.writeFileSync(getEnv('PROFILE_OUTPUT')!, JSON.stringify({ elapsed_ms: Date.now()-start, steps:32, operations:profile }, null, 2))
""")
    subprocess.run([str(Path(args.affon).resolve()), str(runner.relative_to(repo))], cwd=repo, check=True,
                   env={**os.environ, 'PROFILE_MODEL': str(Path(args.model_dir).resolve()), 'PROFILE_OUTPUT': str(output)})
    report = json.loads(output.read_text())
    report.update({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'binary_sha256': hashlib.sha256(Path(args.affon).read_bytes()).hexdigest(),
                   'method': 'One 128-token prefill, then 32 forced single-token steps using token 464; per-call Date.now millisecond wall timing; full logit readback; no_grad is inclusive and must not be summed with primitive operations. Captured graph internals are not attributed.'})
    output.write_text(json.dumps(report, indent=2) + '\n')
finally:
    instrumented.unlink(missing_ok=True)
    runner.unlink(missing_ok=True)
