"""Correlate target thread states with Metal completion events in one xctrace capture."""
import argparse
import xml.etree.ElementTree as E
import json
import collections
import statistics
import bisect
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--export-dir', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--pid', type=int, required=True)
args = p.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
def read(name):
 refs={};cols=[];rows=[]
 def val(e):
  if 'ref' in e.attrib:return refs[e.get('ref')]
  children=[val(c) for c in e]
  if e.tag=='frame':v=e.get('name','?')
  elif e.tag=='backtrace':v=' <- '.join(children)
  elif e.tag=='tagged-backtrace':v=children[0] if children else e.get('fmt','')
  else:v=e.text if e.text and e.text.strip() else e.get('fmt','')
  if 'id' in e.attrib:refs[e.get('id')]=v
  return v
 for _,e in E.iterparse(args.export_dir / (name + '.xml'),events=['end']):
  if e.tag=='schema':cols=[c.findtext('mnemonic') for c in e.findall('col')]
  if e.tag=='row':
   d=dict(zip(cols,[val(c) for c in e]));e.clear()
   if 'process' not in d or d.get('process')==f'affon ({args.pid})':rows.append(d)
 return rows

s=read('metal-application-command-buffer-submissions');c=read('metal-command-buffer-completed');ts=read('thread-state');sc=read('syscall')
main=sorted([x for x in ts if x['thread'].startswith('Main Thread')],key=lambda x:int(x['start']))
ids={x['cmdbuffer-id'] for x in s};cs={x['cmdbuffer-id']:int(x['timestamp']) for x in c if x['cmdbuffer-id'] in ids};lo=min(int(x['start']) for x in s);hi=max(cs.values())
agg=collections.Counter();count=collections.Counter();states=[]
for x in main:
 a=int(x['start']);b=a+int(x['duration']);d=max(0,min(hi,b)-max(lo,a))
 if d:agg[x['state']]+=d;count[x['state']]+=1
 states.append((a,b,x['state']))
assert main, 'Missing main-thread states'
assert abs(sum(agg.values()) - (hi-lo)) <= 1, 'Incomplete or overlapping main-thread coverage'
print('window',lo/1e9,hi/1e9,(hi-lo)/1e9,'sum',sum(agg.values())/1e9)
print('states',dict(agg));print('counts',dict(count))
starts=[a for a,b,t in states];matches=[];where=collections.Counter()
for cid,t in cs.items():
 i=bisect.bisect_right(starts,t)-1
 if i<0 or t>=states[i][1]:where['uncovered']+=1;continue
 a,b,state=states[i];where[state]+=1
 if state!='Blocked':continue
 j=i+1
 while j<len(states) and states[j][2] not in ['Runnable','Running']:j+=1
 if j>=len(states):continue
 runnable=states[j][0];k=j
 while k<len(states) and states[k][2]!='Running':k+=1
 if k>=len(states):continue
 running=states[k][0]
 matches.append(dict(command=cid,completed_ns=t,completed_to_runnable_ns=runnable-t,runnable_to_running_ns=running-runnable,completed_to_running_ns=running-t))
print('completion states',dict(where))
def stats(xs):
 xs=sorted(xs);return dict(count=len(xs),sum_ms=sum(xs)/1e6,median_us=statistics.median(xs)/1e3,p90_us=xs[int(.9*(len(xs)-1))]/1e3)
print('latencies',{k:stats([x[k] for x in matches]) for k in matches[0] if k.endswith('_ns') and k!='completed_ns'})
# Union of main-thread syscall intervals with a waitUntilCompleted stack, clipped to the active window.
waits=[];by_stack=collections.Counter()
for x in sc:
 if x['thread'].startswith('Main Thread') and 'waitUntilCompleted' in x['backtrace']:
  a=max(lo,int(x['start']));b=min(hi,int(x['start'])+int(x['duration']))
  if b>a:waits.append((a,b));by_stack[x['backtrace']]+=b-a
union=[]
for a,b in sorted(waits):
 if union and a<=union[-1][1]:union[-1]=(union[-1][0],max(b,union[-1][1]))
 else:union.append((a,b))
print('waitsyscall union',sum(b-a for a,b in union)/1e6,'calls',len(waits))
summary=dict(submissions=len(s),unmatched_completion_ids=len(ids - set(cs)),window_ns=[lo,hi],window_ms=(hi-lo)/1e6,states_ms={k:v/1e6 for k,v in agg.items()},state_counts=dict(count),matched_completion_events=len(cs),completion_states=dict(where),blocked_completion_latencies={k:stats([x[k] for x in matches]) for k in matches[0] if k.endswith('_ns') and k!='completed_ns'},wait_syscalls=dict(count=len(waits),union_ms=sum(b-a for a,b in union)/1e6),top_wait_stacks=[dict(stack=k,ms=v/1e6) for k,v in by_stack.most_common(5)])
json.dump(summary,open(args.output_dir / 'summary.json','w'),indent=2)
json.dump(matches[:20],open(args.output_dir / 'example-wakeups.json','w'),indent=2)
