"""Collect the bounded telemetry ring continuously; fail if any records are lost."""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--affon', required=True)
p.add_argument('--output', required=True)
p.add_argument('--port', type=int, default=8767)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
result = out / 'metrics.json'
result.unlink(missing_ok=True)
env = {**os.environ, 'RUNTIME_TELEMETRY_CONSOLE':'1', 'RUNTIME_TELEMETRY_CONSOLE_PORT':str(a.port), 'PROFILE_OUTPUT':str(result.resolve()), 'AFFON_DEVICE':'metal'}
records = []; cursor = 0; missed = False
with (out / 'run.log').open('w') as log:
 proc = subprocess.Popen([a.affon, 'apps/hf-inference/onnx/telemetry-whisper.ts'], env=env, stdout=log, stderr=log)
 try:
  deadline = time.monotonic()+90
  while time.monotonic()<deadline:
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/api/traces?since={cursor}', timeout=2) as response: batch=json.load(response)
    records.extend(batch['records']); cursor=batch['next_cursor']; missed |= batch['missed']
    if result.exists():
     try: json.loads(result.read_text())
     except (ValueError, OSError): pass
     else: break
   except OSError:
    if proc.poll() is not None: raise RuntimeError('Runner exited; inspect run.log')
   time.sleep(.002)
  else: raise RuntimeError('Capture timed out')
 finally:
  proc.terminate(); proc.wait()
(out/'traces.json').write_text(json.dumps({'missed':missed,'records':records}))
starts={}; groups={}; matrices={}
phase_starts={}; windows=[]
for r in records:
 if r['name'] != 'hf.whisper.decoder': continue
 if r['kind']=='span_start': phase_starts[r['span_id']]=r['timestamp_ns']
 elif r['kind']=='span_end' and r['span_id'] in phase_starts: windows.append((phase_starts[r['span_id']],r['timestamp_ns']))
for r in records:
 key=(r['trace_id'],r['span_id'])
 if r['kind']=='span_start': starts[key]=r
 elif r['kind']=='event' and r['name']=='matmul_shape' and key in starts:
  starts[key]['matrix']=r['attributes']
 elif r['kind']=='span_end' and key in starts:
  s=starts.pop(key)
  if 'op' not in s['attributes']: continue
  if not any(lo <= s['timestamp_ns'] and r['timestamp_ns'] <= hi for lo, hi in windows): continue
  label=s['attributes']['op']; g=groups.setdefault(label,{'calls':0,'ms':0})
  g['calls']+=1; g['ms']+=(r['timestamp_ns']-s['timestamp_ns'])/1e6
  if 'matrix' in s:
   matrix=json.dumps(s['matrix'],sort_keys=True); bucket=matrices.setdefault(matrix,{'calls':0,'ms':0});bucket['calls']+=1;bucket['ms']+=(r['timestamp_ns']-s['timestamp_ns'])/1e6
report={'binary_sha256':hashlib.sha256(Path(a.affon).read_bytes()).hexdigest(),'method':'Native compute span durations inside hf.whisper.decoder time windows; one reference transcription including prefix; no warmup; complete ring collection required. CPU wall time includes synchronous GPU waits, not GPU-only kernel time.','missed':missed,'records':len(records),'decoder_steps':len(windows),'decoder_ms':sum((hi-lo)/1e6 for lo,hi in windows),'matrices':dict(sorted(matrices.items(),key=lambda kv:-kv[1]['ms'])),'operations':dict(sorted(groups.items(), key=lambda kv:-kv[1]['ms']))}
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if missed: raise RuntimeError('Trace records lost; timing totals are incomplete')
