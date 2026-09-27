"""Shared CPU reference generation and synchronized PyTorch measurement worker."""
import argparse
import json
import time
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from cases import cases


def evaluate(c, t):
    a, b = t['a'], t.get('b')
    if c.get('narrow_inputs'):
        a=a[1:-1,1:-1]
        if b is not None: b=b[1:-1,1:-1]
    if c.get('transpose_a'): a = a.transpose(0, 1)
    if c.get('transpose_b'): b = b.transpose(-2, -1)
    if c['kind'] == 'sequence':
        for _ in range(c['length']):
            if c.get('fused'):
                a = a @ b + t['bias']
                if c['fused'] == 'gelu': a = F.gelu(a, approximate='tanh')
            else: a = F.relu(a @ b) if c['op'] == 'matmul' else F.relu(a) + b
        return a
    op = c['op']
    if op == 'layer_norm':
        moved = a.movedim(c['axis'], -1)
        return F.layer_norm(moved, (moved.shape[-1],), eps=1e-5).movedim(-1, c['axis'])
    if op == 'softmax': return a.softmax(dim=c['axis'])
    if op in ['sum_axis', 'mean_axis']: return getattr(a, op.split('_')[0])(dim=c['axis'])
    if op in ['sum_all', 'mean_all']: return getattr(a, op.split('_')[0])().reshape(1)
    if op.split('_')[0] in ['min','max','variance','std','argmin','argmax']:
        name = op.split('_')[0]
        name = {'min':'amin','max':'amax','variance':'var'}.get(name,name)
        kwargs = {'correction':0} if name in ['var','std'] else {}
        if c.get('axis') is not None: kwargs['dim']=c['axis']
        result = getattr(torch,name)(a,**kwargs)
        return result.reshape(1) if c.get('axis') is None else result
    if op == 'gather': return torch.gather(a,c['axis'],t['index'])
    if op == 'index_select': return torch.index_select(a,c['axis'],t['index'])
    if op == 'reshape': return a.reshape(c['target'])
    if op == 'contiguous': return a.contiguous()
    if op == 'permute': return a.permute(c['axes'])
    if op == 'slice': return a[1:-1,2:-1:2]
    if op == 'squeeze': return a.squeeze()
    if op == 'unsqueeze': return a.unsqueeze(c['axis'])
    if op == 'cat': return torch.cat([a,b if b is not None else a],dim=c['axis'])
    if op == 'stack': return torch.stack([a,b if b is not None else a],dim=c['axis'])
    if op == 'silu': return F.silu(a)
    if op == 'gelu': return F.gelu(a, approximate='tanh')
    return getattr(torch, op)(a, b) if b is not None else getattr(torch, op)(a)


def data(shape, offset):
    count = 1
    for d in shape: count *= d
    return (((torch.arange(count, dtype=torch.int64)*7+offset)%31-15).float()/32).reshape(shape)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'measure'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', choices=['cpu','mps'], default='mps')
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--view-repeats', type=int, default=None)
    p.add_argument('--filter', default='.*', help='Regular expression selecting case IDs')
    p.add_argument('--warmup', type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    sync = torch.mps.synchronize if args.device == 'mps' else lambda: None
    args.root.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if args.action == 'prepare':
            entries = [c for c in cases() if re.search(args.filter,c['id'])]
            if not entries: raise ValueError('Case filter matched nothing')
            for c in entries:
                if args.device in c.get('unsupported_devices',{}): c['unsupported']=c['unsupported_devices'][args.device]
            excluded = [c for c in entries if c.get('unsupported')]
            entries = [c for c in entries if not c.get('unsupported')]
            if not entries: raise ValueError('No supported cases selected')
            for c in entries:
                t = {'a': data(c['a'], 1)}
                if c.get('b'): t['b'] = data(c['b'], 9)
                if c['op'] in ['log','sqrt']: t['a'] = t['a'].abs()+0.25
                if c['op'] == 'div' or c['kind'] == 'sequence': t['b'] = t['b'].abs()+0.25
                if c['kind'] == 'sequence' and c['op'] == 'matmul':
                    width = c['b'][0]
                    t['b'] = torch.eye(width)*0.5 + torch.ones((width,width))*(0.5/width)
                if c.get('fused'): t['bias'] = data(c['a'],17)*0.1+0.05
                if c.get('index'):
                    count = 1
                    for dim in c['index']: count *= dim
                    t['index'] = (((torch.arange(count,dtype=torch.int64)%5)*7+3)%c['index_bound']).reshape(c['index'])
                out = evaluate(c,t)
                c['input_dtypes'] = {k:str(v.dtype) for k,v in t.items()}
                c['output_dtype'] = str(out.dtype)
                c['expected'] = out.flatten().tolist()
                c['output_shape'] = list(out.shape)
                save_file(t,str(args.root/(c['id']+'.safetensors')))
            config = dict(schema=1, device=args.device, dtype='float32', atol=1e-4, rtol=1e-4, samples=args.samples,
                          repeats=args.repeats, view_repeats=args.view_repeats or args.repeats, warmup=args.warmup, cases=entries, excluded_cases=excluded)
            (args.root/'manifest.json').write_text(json.dumps(config))
            return
        config = json.loads((args.root/'manifest.json').read_text())
        results = []
        for c in config['cases']:
            t = {k:v.to(args.device) for k,v in load_file(str(args.root/(c['id']+'.safetensors'))).items()}
            if c.get('narrow_inputs'):
                t['a']=t['a'][1:-1,1:-1]
                if 'b' in t: t['b']=t['b'][1:-1,1:-1]
            if c.get('transpose_a'): t['a'] = t['a'].transpose(0,1)
            if c.get('transpose_b'): t['b'] = t['b'].transpose(-2,-1)
            prepared = dict(c,transpose_a=False,transpose_b=False,narrow_inputs=False)
            repeats = config['view_repeats'] if c.get('completion')=='host-view' else config['repeats']
            # Eager sequence mode matches Affon's per-operation completion.
            # Scoped mode lets PyTorch queue the entire sequence, then completes.
            def run():
                if c['kind'] == 'sequence' and c['mode'] == 'eager':
                    out = t['a']
                    for _ in range(c['length']):
                        if c.get('fused'):
                            out = out @ t['b'] + t['bias']
                            if c['fused'] == 'gelu': out = F.gelu(out, approximate='tanh')
                            sync()
                        elif c['op'] == 'matmul':
                            out = out @ t['b']; sync()
                            out = F.relu(out); sync()
                        else:
                            out = F.relu(out); sync()
                            out = out+t['b']; sync()
                    return out
                out = evaluate(prepared,t)
                if c.get('torch_completion',c.get('completion')) != 'host-view': sync()
                return out
            out = run().cpu()
            reference = torch.tensor(c['expected'],dtype=out.dtype).reshape(c['output_shape'])
            torch.testing.assert_close(out, reference, atol=config['atol'], rtol=config['rtol'])
            for _ in range(config['warmup']): run()
            samples = []
            for _ in range(config['samples']):
                sync(); start = time.perf_counter_ns()
                for _ in range(repeats): out = run()
                samples.append((time.perf_counter_ns()-start)/1e6/repeats)
            results.append(dict(id=c['id'], samples_ms=samples, correct=True,
                                max_absolute_error=float((out.cpu()-reference).abs().max()), implementation='PyTorch automatic'))
        args.output.write_text(json.dumps(dict(torch=torch.__version__, cpu_threads=1, device=args.device, results=results), indent=2)+'\n')

if __name__ == '__main__': main()
