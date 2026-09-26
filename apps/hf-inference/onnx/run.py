"""Sequential native adapter/graph comparison; model preparation is separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--affon',required=True);p.add_argument('--model-dir',required=True)
p.add_argument('--graph-dir',required=True);p.add_argument('--output',required=True)
p.add_argument('--routes',nargs='+',choices=['graph','adapter','hf-onnx'],default=['graph','adapter'])
p.add_argument('--iterations',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
p.add_argument('--build-profile',default='unspecified')
a=p.parse_args();repo=Path(__file__).resolve().parents[3];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
binary=Path(a.affon).resolve()
meta={'platform':platform.platform(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'build_profile':a.build_profile,'iterations':a.iterations,'warmups':a.warmups,
      'graph_sha256':hashlib.sha256((Path(a.graph_dir)/'graph.json').read_bytes()).hexdigest(),
      'source_sha256':{str(n.relative_to(repo)):hashlib.sha256(n.read_bytes()).hexdigest() for n in [repo/'packages/@affon/onnx/tools/convert.py',repo/'packages/@affon/onnx/src/runtime.ts',repo/'packages/@affon/onnx/src/spatial.ts',repo/'packages/@affon/huggingface/src/onnx.ts',Path(__file__).with_name('audit.ts')]},'runs':[]}
for device in ['cpu','metal']:
    for route in a.routes:
        name=f'{route}-{device}';print('Starting',name,flush=True)
        command=[str(binary),'apps/hf-inference/onnx/audit.ts']
        if platform.system()=='Darwin':command=['/usr/bin/time','-l',*command]
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=repo,stdout=log,stderr=log,env={**os.environ,
                'AFFON_DEVICE':device,'AFFON_ONNX_ROUTE':route,'AFFON_ONNX_ITERATIONS':str(a.iterations),'AFFON_ONNX_WARMUPS':str(a.warmups),'AFFON_ONNX_DIR':str(Path(a.graph_dir).resolve()),
                'AFFON_HF_MODEL_DIR':str(Path(a.model_dir).resolve()),'AFFON_HF_REPORT':str(out/(name+'.json'))})
        if result.returncode:raise RuntimeError(f'{name} failed; inspect log/report')
        peak=re.search(r'(\d+)\s+maximum resident set size',(out/(name+'.log')).read_text())
        meta['runs'].append({'name':name,'os_peak_rss_bytes':int(peak[1]) if peak else None})
        (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
