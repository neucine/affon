"""Join xctrace Metal tables by command ID; report observed intervals, not causal costs."""
import argparse
import xml.etree.ElementTree as E
import json
import statistics
import collections
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--export-dir', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--pid', type=int, required=True)
p.add_argument('--start', type=float, default=5)
p.add_argument('--end', type=float, default=16)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
def read(name):
 refs={}; cols=[];rows=[]
 def val(e):
  if 'ref' in e.attrib:return refs[e.get('ref')]
  for c in e:val(c)
  v=e.text if e.text and e.text.strip() else e.get('fmt','')
  if 'id' in e.attrib:refs[e.get('id')]=v
  return v
 for _,e in E.iterparse(args.export_dir / (name + '.xml'),events=['end']):
  if e.tag=='schema':cols=[c.findtext('mnemonic') for c in e.findall('col')]
  if e.tag=='row':
   d=dict(zip(cols,[val(c) for c in e]));e.clear()
   if 'process' not in d or d['process']==f'affon ({args.pid})':rows.append(d)
 return rows
s=read('metal-application-command-buffer-submissions')
g=read('metal-gpu-intervals')
c=read('metal-command-buffer-completed')
groups=collections.defaultdict(list)
for x in g:groups[x['cmdbuffer-id']].append(x)
complete={x['cmdbuffer-id']:int(x['timestamp']) for x in c}
assert len({x['cmdbuffer-id'] for x in s}) == len(s), 'Duplicate submissions'
assert all(int(a['start']) <= int(b['start']) for a,b in zip(s,s[1:])), 'Unordered submissions'
rows=[]
missing=0
for a,b in zip(s,s[1:]):
 key=a['cmdbuffer-id'];start=int(a['start']);end=start+int(a['duration'])
 if start<args.start*1e9 or start>=args.end*1e9:continue
 if key not in groups or key not in complete or b['cmdbuffer-id'] not in groups:
  missing+=1
  continue
 gs=min(int(x['start']) for x in groups[key]);ge=max(int(x['start'])+int(x['duration']) for x in groups[key]);ce=complete[key];nxt=int(b['start'])
 rows.append(dict(command=key,start_ns=start,encoding_ns=end-start,encoding_end_to_gpu_ns=gs-end,gpu_span_ns=ge-gs,gpu_end_to_completed_ns=ce-ge,completed_to_next_encoding_ns=nxt-ce,gpu_end_to_next_gpu_ns=min(int(x['start']) for x in groups[b['cmdbuffer-id']])-ge))
anomalies = [x for x in rows if any(v < 0 for k, v in x.items() if k.endswith('_ns'))]
matched_count = len(rows)
rows = [x for x in rows if x not in anomalies]
summary={'window_seconds':[args.start,args.end],'unmatched_commands':missing,'matched_commands':matched_count,'excluded_anomalies':anomalies,'commands':len(rows),'metrics':{}}
for key in rows[0]:
 if not key.endswith('_ns') or key=='start_ns':continue
 xs=sorted(x[key] for x in rows)
 summary['metrics'][key]={'sum_ms':sum(xs)/1e6,'median_us':statistics.median(xs)/1000,'p90_us':xs[int(.9*(len(xs)-1))]/1000,'negative_count':sum(x<0 for x in xs)}
print(json.dumps(summary,indent=2))
json.dump(summary,open(args.output_dir / 'summary.json','w'),indent=2)
json.dump(rows[:20],open(args.output_dir / 'example-commands.json','w'),indent=2)
