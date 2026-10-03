"""Reproducible isolated CPU phases; optional macOS native stack samples."""
import argparse, collections, hashlib, json, os, pathlib, platform, statistics, subprocess, tempfile, time, resource
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', required=True); p.add_argument('--affon', required=True)
p.add_argument('--output', required=True); p.add_argument('--threads', type=int, default=1)
p.add_argument('--batch',type=int,default=10)
p.add_argument('--runs',type=int,default=30); p.add_argument('--sample',action='store_true')
a=p.parse_args();
if a.runs < 1 or a.batch < 1 or a.threads < 1: p.error('runs, batch and threads must be positive')
app=pathlib.Path(__file__).resolve().parent; root=app.parents[1]
out=pathlib.Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
model=pathlib.Path(a.model_dir).resolve(); binary=pathlib.Path(a.affon).resolve()
env=os.environ|{'MODEL_DIR':str(model),'AFFON_DEVICE':'cpu','PROFILE_RUNS':str(a.runs),'PROFILE_BATCH':str(a.batch),'PROFILE_OUTPUT':str(out/'affon.json'),'RUNTIME_PACKAGE_PATH':str(root/'packages'),'AFFON_CPU_THREADS':str(a.threads),'ORT_THREADS':str(a.threads),'VECLIB_MAXIMUM_THREADS':str(a.threads),'OMP_NUM_THREADS':str(a.threads),'OPENBLAS_NUM_THREADS':str(a.threads)}
with tempfile.TemporaryDirectory(prefix='affon-cpu-audit-') as temp:
    before_usage=resource.getrusage(resource.RUSAGE_CHILDREN); tick=time.perf_counter()
    subprocess.run([str(binary),str(app/'profile.ts')],env=env,cwd=temp,check=True)
    wall=time.perf_counter()-tick; after_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    process_usage={'wall_seconds':wall,'user_seconds':after_usage.ru_utime-before_usage.ru_utime,'system_seconds':after_usage.ru_stime-before_usage.ru_stime,'peak_rss_bytes':after_usage.ru_maxrss*(1 if platform.system()=='Darwin' else 1024)}
    process_usage['mean_cpu_percent']=100*(process_usage['user_seconds']+process_usage['system_seconds'])/wall
    if a.sample:
        for mode in ['graph','preprocess']:
            with (out/f'{mode}.log').open('w') as log:
                proc=subprocess.Popen([str(binary),str(app/'profile.ts')],env=env|{'PROFILE_MODE':mode,'PROFILE_RUNS':'100000'},cwd=temp,stdout=log,stderr=log)
                try:
                    time.sleep(3)
                    subprocess.run(['sample',str(proc.pid),'5','1','-file',str(out/f'{mode}-stacks.txt')],check=True)
                finally:
                    proc.terminate(); proc.wait(timeout=10)
    # Set thread limits before importing NumPy/ORT, and contain ORT telemetry files.
    os.environ.update(env); os.chdir(temp)
    import importlib.util, numpy as np
    spec=importlib.util.spec_from_file_location('baseline',app/'baseline.py'); base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    rgb=np.fromfunction(lambda y,x,c:(x*3+y*5+c*47)%256,(260,320,3),dtype=int).astype(np.uint8)
    pixels=base.preprocess(rgb)
    for _ in range(5): base.session.run(None,{'pixels':pixels})
    times={'graph_ms':[],'preprocessing_ms':[]}
    for _ in range(a.runs):
        t=time.perf_counter_ns()
        for _ in range(a.batch): expected=base.session.run(None,{'pixels':pixels})[0]
        times['graph_ms'].append((time.perf_counter_ns()-t)/1e6/a.batch)
        t=time.perf_counter_ns()
        for _ in range(a.batch): base.preprocess(rgb)
        times['preprocessing_ms'].append((time.perf_counter_ns()-t)/1e6/a.batch)
    native=json.loads((out/'affon.json').read_text()); actual=np.array(native['logits'])
    error=float(abs(actual-expected).max()); parity=bool(np.allclose(actual,expected,atol=1e-4,rtol=1e-4))
    ops=collections.Counter(); nodes=collections.Counter()
    for e in native['operators']: ops[e['op']]+=e['elapsed_ms']/10; nodes[e['name']]+=e['elapsed_ms']/10
    def stats(xs): return {'median':statistics.median(xs),'p95':float(np.percentile(xs,95)),'min':min(xs),'max':max(xs)}
    before={m['scope']+'.'+m['name']:m['value'] for m in native['before']}
    delta={m['scope']+'.'+m['name']:(m['value']-before.get(m['scope']+'.'+m['name'],0))/(a.runs*a.batch) for m in native['after'] if m['kind']=='counter'}
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for d in [root/'packages/@affon/onnx/src',root/'packages/@affon/huggingface/src/processors'] for f in d.rglob('*.ts')}
    report={'platform':platform.platform(),'native_process_usage':process_usage,'threads_requested':a.threads,'runs':a.runs,'batch':a.batch,'warmups':5,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((model/'model.onnx').read_bytes()).hexdigest(),'source_hashes':hashes,'affon':{k:stats(native[k]) for k in ['graph_ms','readback_ms','preprocessing_ms']},'onnxruntime':{k:stats(v) for k,v in times.items()},'ort_samples':times,'parity':parity,'max_absolute_error':error,'operator_mean_ms':dict(ops.most_common()),'top_nodes_mean_ms':nodes.most_common(15),'counters_per_iteration':delta,'ort_version':base.ort.__version__}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','ort_samples']},indent=2))
    if not parity: raise AssertionError('Logits differ')
