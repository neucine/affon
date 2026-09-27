#!/usr/bin/env python3
"""Run curated compute benchmarks, expose OpTag coverage, optionally gate regressions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    source = path.read_text()
    body = source.split('pub const OpTag = enum {', 1)[1].split('};', 1)[0]
    return re.findall(r'^\s*([a-z][a-z0-9_]*)\s*,', body, re.M)


def stats(samples):
    return dict(median_ms=statistics.median(samples), p90_ms=sorted(samples)[math.ceil(len(samples)*.9)-1], samples_ms=samples)


def case_dimensions(c):
    shapes = {k:c[k] for k in ['a','b','index'] if k in c}
    if c.get('fused'): shapes['bias']=c['a']
    logical = {k:list(v) for k,v in shapes.items()}
    layouts = {k:'contiguous' for k in shapes}
    for key in ['a','b']:
        if c.get('narrow_inputs') and key in logical:
            logical[key] = [dim-2 for dim in logical[key]]
            layouts[key] = 'offset view'
        if c.get('transpose_'+key):
            i,j = (0,1) if key=='a' else (-2,-1)
            logical[key][i],logical[key][j] = logical[key][j],logical[key][i]
            layouts[key] = 'transposed offset view' if c.get('narrow_inputs') else 'transposed view'
    return dict(input_shapes=logical, fixture_shapes=shapes, input_layouts=layouts,
                input_dtypes=c.get('input_dtypes',{}), output_dtype=c.get('output_dtype'),
                output_shape=c.get('output_shape'), execution=c['mode'], kind=c['kind'], completion=dict(affon=c.get('completion','device-complete'),torch=c.get('torch_completion',c.get('completion','device-complete'))),
                parameters={k:c[k] for k in ['axis','axes','target','length','operations','fused','narrow_inputs'] if k in c})


def ranked_gaps(rows):
    # View-only API costs must not be mistaken for GPU kernel latency.
    eligible = [r for r in rows if r['correct'] and r['timing_resolved']]
    return [dict(id=r['id'], kind=r.get('kind','operator'), family=r['family'], ratio=r['ratio'],
                 excess_ms=r['affon']['median_ms']-r['torch']['median_ms'])
            for r in sorted(eligible,key=lambda r:r['affon']['median_ms']-r['torch']['median_ms'],reverse=True)
            if r['ratio']>1]


def compare_baseline(current, baseline, tolerance):
    for key in ['suite_hash', 'hardware', 'platform', 'device', 'torch', 'timing', 'compute_op_tags_sha256']:
        if current['environment'][key] != baseline['environment'][key]:
            raise ValueError(f'Incompatible baseline: {key}')
    old = {r['id']:r for r in baseline['results']}
    if set(old) != {r['id'] for r in current['results']}: raise ValueError('Incompatible baseline: case IDs')
    regressions = []
    for row in current['results']:
        prior = old[row['id']]
        if not row['timing_resolved'] or not prior['timing_resolved']:
            raise ValueError(f'Unresolved timer samples: {row["id"]}; increase repeats')
        if not row['correct'] or not prior['correct']: raise ValueError('Incorrect results cannot be a performance baseline')
        ratio = row['affon']['median_ms']/prior['affon']['median_ms']
        if ratio > 1+tolerance: regressions.append(dict(id=row['id'], ratio=ratio))
    return regressions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--affon', type=Path, required=True)
    p.add_argument('--compute', type=Path, required=True, help='Compute repository for live OpTag coverage')
    p.add_argument('--python', default=sys.executable, help='Python with torch and safetensors installed')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','mps'], default='mps')
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--view-repeats', type=int, help='Repeat count for cheap host-view operations (defaults to --repeats)')
    p.add_argument('--filter', default='.*', help='Regex selecting case IDs; unmatched OpTags remain uncovered')
    p.add_argument('--baseline', type=Path, help='Previous summary.json on the same host/suite')
    p.add_argument('--regression-tolerance', type=float, default=.15)
    args = p.parse_args()
    if args.view_repeats is None: args.view_repeats=args.repeats
    if min(args.view_repeats,args.samples,args.repeats,args.warmup)<1 or args.regression_tolerance<0: p.error('Counts must be positive and tolerance nonnegative')
    out=args.output.resolve()
    if out.exists(): p.error('Output directory already exists; use a fresh directory to preserve evidence')
    out.mkdir(parents=True)
    fixture=out/'fixtures'
    binary=args.affon.resolve()
    env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0', AFFON_DEVICE='metal' if args.device=='mps' else 'cpu')
    # Diagnostic forcing/instrumentation would invalidate automatic-dispatch timing.
    for key in ['AFFON_METAL_COMMAND_TIMING','AFFON_METAL_MATMUL','AFFON_METAL_LAYOUT_MPS','AFFON_METAL_DISPATCH_TRACE','COMPUTE_BENCH_DISPATCH_ONLY']:
        env.pop(key,None)
    def call(name, command, extra=None):
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=REPO,env=env|(extra or {}),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name, 'passed', flush=True)
    call('prepare',[args.python,str(HERE/'torch_worker.py'),'prepare','--root',str(fixture),'--device',args.device,'--samples',str(args.samples),'--repeats',str(args.repeats),'--warmup',str(args.warmup),'--filter',args.filter,'--view-repeats',str(args.view_repeats)])
    config=json.loads((fixture/'manifest.json').read_text())
    call('dispatch',[str(binary),'run',str(HERE/'affon.ts')],{
        'COMPUTE_BENCH_FIXTURES':str(fixture),'COMPUTE_BENCH_OUTPUT':str(out/'dispatch.json'),
        'COMPUTE_BENCH_DISPATCH_ONLY':'1','AFFON_METAL_DISPATCH_TRACE':'1'})
    dispatch={r['id']:r for r in json.loads((out/'dispatch.json').read_text())['results']}
    measured={}
    # ABBA, each entry a fresh process, no concurrent benchmark jobs.
    order=['affon-a','torch-a','torch-b','affon-b']
    for name in order:
        if name.startswith('affon'):
            call(name,[str(binary),'run',str(HERE/'affon.ts')],{'COMPUTE_BENCH_FIXTURES':str(fixture),'COMPUTE_BENCH_OUTPUT':str(out/(name+'.json'))})
        else:
            call(name,[args.python,str(HERE/'torch_worker.py'),'measure','--root',str(fixture),'--device',args.device,'--output',str(out/(name+'.json'))])
        measured[name]=json.loads((out/(name+'.json')).read_text())
    rows=[]
    lookup={k:{r['id']:r for r in v['results']} for k,v in measured.items()}
    for c in config['cases']:
        a=[lookup[n][c['id']] for n in ['affon-a','affon-b']]
        t=[lookup[n][c['id']] for n in ['torch-a','torch-b']]
        a_stats=stats(sum([r['samples_ms'] for r in a],[])); t_stats=stats(sum([r['samples_ms'] for r in t],[]))
        resolved=all(ms>=10 for r in a for ms in r['sample_totals_ms'])
        rows.append(dict(id=c['id'],kind=c['kind'],op=c['op'],family=c['family'],mode=c['mode'],
                         correct=all(r['correct'] for r in a+t) and dispatch[c['id']]['correct'],timing_resolved=resolved,
                         implementation=dispatch[c['id']]['implementation'],dispatch=dispatch[c['id']]['dispatch'],
                         scope_stats=dispatch[c['id']].get('scope_stats'), dimensions=case_dimensions(c),
                         repeats=config['view_repeats'] if c.get('completion')=='host-view' else config['repeats'],
                         affon=a_stats,torch=t_stats,ratio=a_stats['median_ms']/t_stats['median_ms'] if resolved else None,
                         affon_group_medians_ms=[statistics.median(r['samples_ms']) for r in a],
                         torch_group_medians_ms=[statistics.median(r['samples_ms']) for r in t]))
    tags=inventory(args.compute/'src/shared/types/operation/tag.zig')
    covered={c['op'] for c in config['cases'] if c['kind']=='operator'}
    if covered-set(tags): raise ValueError(f'Cases reference unknown OpTags: {covered-set(tags)}')
    coverage={tag:dict(status='measured' if tag in covered else 'uncovered',cases=[c['id'] for c in config['cases'] if c['kind']=='operator' and c['op']==tag]) for tag in tags}
    for tag, entry in coverage.items():
        entry['variants'] = [dict(id=r['id'], **r['dimensions'], implementation=r['implementation'], observed_events=r['dispatch']) for r in rows if r['kind']=='operator' and r['op']==tag]
        entry['unsupported_cases'] = [c for c in config.get('excluded_cases',[]) if c['op']==tag]
    hardware=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor()
    suite_hash=hashlib.sha256(''.join(sha(HERE/f) for f in ['run.py','cases.py','torch_worker.py','affon.ts']).encode()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    summary=dict(schema=2,environment=dict(suite_hash=suite_hash,hardware=hardware,platform=platform.platform(),device=args.device,
                 torch=measured['torch-a']['torch'],binary_sha256=sha(binary),compute_op_tags_sha256=sha(args.compute/'src/shared/types/operation/tag.zig'),
                 timing=dict(warmup=args.warmup,repeats=args.repeats,view_repeats=args.view_repeats,samples_per_process=args.samples,process_order=order,completion='device operations complete per operator / per scoped sequence; host views need no device fence',clock='Affon Date.now milliseconds; PyTorch perf_counter_ns')),
                 coverage=dict(total=len(tags),measured=len(covered),operations=coverage,implementation_coverage='Native ordinary f32 matmul dispatch and compiled epilogue fusion events; other routes unobserved',dtype_coverage=sorted({dtype for c in config['cases'] for dtype in c.get('input_dtypes',{}).values()}), output_dtype_coverage=sorted({c['output_dtype'] for c in config['cases']}),execution_coverage=['eager operators','dependent chains eager/scoped'], unsupported_cases=config.get('excluded_cases',[])),results=rows, ranked_gaps=ranked_gaps(rows))
    if args.baseline:
        summary['regressions']=compare_baseline(summary,json.loads(args.baseline.read_text()),args.regression_tolerance)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Compute benchmark coverage','',f'{len(covered)}/{len(tags)} OpTags have measured operator cases. Uncovered entries remain visible in summary.json.',
           '', 'Automatic dispatch only; this does not establish coverage of every implementation, dtype or shape. Ratios above 1 mean Affon is slower.',
           '', '| Case | Affon median ms | PyTorch median ms | Affon / PyTorch | Dispatch events (separate diagnostic) |', '|---|---:|---:|---:|---|']
    for row in rows:
        ratio=f'{row["ratio"]:.2f}×' if row['ratio'] is not None else 'timer unresolved'
        events=', '.join(name.removeprefix('metal_dispatch_') for name in row['dispatch']) or 'not instrumented'
        lines.append(f'| {row["id"]} | {row["affon"]["median_ms"]:.4f} | {row["torch"]["median_ms"]:.4f} | {ratio} | {events} |')
    lines+=['','p90 values describe repeat-block averages, not individual-call tail latency.','', 'Uncovered: '+', '.join(t for t in tags if t not in covered)+'.']
    for kind in ['operator','sequence']:
        lines += ['', f'## Largest measured {kind} gaps by excess milliseconds', '', '| Case | Excess ms | Affon / PyTorch |', '|---|---:|---:|']
        for gap in [g for g in summary['ranked_gaps'] if g['kind']==kind][:15]:
            lines.append(f"| {gap['id']} | {gap['excess_ms']:.4f} | {gap['ratio']:.2f}× |")
    lines += ['', 'Unresolved timers are excluded from rankings. Layout views measure API/view construction costs, not GPU kernel execution.', '', 'Unsupported selected cases:']
    lines += [f"- {c['id']}: {c['unsupported']}" for c in config.get('excluded_cases',[])]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(rows)} cases passed; {len(covered)}/{len(tags)} operations measured; report: {out}/README.md')
    if summary.get('regressions'): raise SystemExit(2)

if __name__=='__main__': main()
