#!/usr/bin/env python3
"""Profile both engines with the same public Metal command-buffer timestamps.
Uses a profiling-only native interposer; GPU callbacks add some host overhead,
so wall-time benchmarks run separately without it. No tensor data is intercepted.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=['affon','mlx'],required=True)
    p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--m128-only',action='store_true')
    p.add_argument('--warmup-ms',type=int,default=300)
    p.add_argument('--samples',type=int,default=100)
    args=p.parse_args()
    cases=[128] if args.m128_only else [1,8,32,128]
    with tempfile.TemporaryDirectory(prefix='affon-metal-timestamps-') as folder:
        folder=Path(folder); lib=folder/'timestamps.dylib'; data=folder/'timestamps.jsonl'
        subprocess.run(['xcrun','clang','-dynamiclib','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',
                        str(root/'tools/metal-command-timestamps.m'),'-o',str(lib)],check=True)
        env=dict(os.environ,AFFON_GPU_PROFILE_PATH=str(data),DYLD_INSERT_LIBRARIES=str(lib),AFFON_QMM_WARMUP_MS=str(args.warmup_ms),AFFON_QMM_SAMPLES=str(args.samples),AFFON_QMM_M128_ONLY='1' if args.m128_only else '0')
        script=root/('test/benchmark/quantized/trace-quantized-metal.ts' if args.engine=='affon' else 'tools/trace-mlx-quantized.py')
        subprocess.run([str(args.executable.absolute()),str(script)],env=env,check=True)
        events=sorted((json.loads(line) for line in data.read_text().splitlines()),key=lambda e:e['commit_ns'])
        groups=[]
        for event in events:
            if not groups or event['commit_ns']-groups[-1][-1]['commit_ns']>80_000_000:groups.append([])
            groups[-1].append(event)
        groups=[g for g in groups if len(g)>=args.samples]
        if len(groups)!=len(cases):raise RuntimeError(f'Expected complete workloads, got {[len(g) for g in groups]}')
        results=[]
        for m,g in zip(cases,groups):
            times=[e['gpu_ns']/1e6 for e in g[-args.samples:]]
            results.append(dict(m=m,n=2048,k=1024,bits=8,group_size=64,samples_ms=times,median_gpu_ms=statistics.median(times),observed_command_count=len(g)))
    report=dict(engine=args.engine,warmup_ms=args.warmup_ms,samples=args.samples,scope='MTLCommandBuffer.GPUEndTime - GPUStartTime; same interposer/API on both engines; full QMM evaluated',
                caveat='GPU command work including encoding-related device overhead and any MLX casts; excludes host encoding/queue wait/readback; profiling callback host costs excluded from separate wall benchmark',results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    for r in results:print(f"{args.engine} M={r['m']} GPU command {r['median_gpu_ms']:.6f}ms")


if __name__=='__main__':main()
