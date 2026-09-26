"""Bounded ONNX opset-17 static-f32 graph conversion, independent of model architecture."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper
from onnx.reference import ReferenceEvaluator
from safetensors.numpy import save_file

SUPPORTED = {'Slice','Unsqueeze','Conv','Reshape','Transpose','Concat','Add','Mul','Div','MatMul','LayerNormalization','Softmax','Cast','Erf','Gather','Gemm','Identity','Pad','Clip','GlobalAveragePool','Flatten'}
ATTRS = {'Pad':{'mode'}, 'Flatten':{'axis'}, 'Conv':{'auto_pad','dilations','group','kernel_shape','pads','strides'}, 'Reshape':{'allowzero'},
 'Transpose':{'perm'}, 'Concat':{'axis'}, 'LayerNormalization':{'axis','epsilon','stash_type'},
 'Softmax':{'axis'}, 'Cast':{'to'}, 'Gather':{'axis'}, 'Gemm':{'alpha','beta','transA','transB'}}

def convert(source, output):
    model = onnx.load(source)
    onnx.checker.check_model(model)
    if [(x.domain,x.version) for x in model.opset_import] != [('',17)]:
        raise ValueError('Only default-domain opset 17 is supported')
    if model.functions or model.graph.sparse_initializer:
        raise ValueError('Functions and sparse initializers are unsupported')
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    info = {}
    for value in [*model.graph.input,*model.graph.value_info,*model.graph.output]:
        t = value.type.tensor_type
        dims = [d.dim_value if d.HasField('dim_value') else None for d in t.shape.dim]
        if all(d is not None and d > 0 for d in dims): info[value.name] = {'shape':dims,'dtype':t.elem_type}
    constants = {t.name:numpy_helper.to_array(t) for t in model.graph.initializer}
    if any(v.name in constants for v in model.graph.input): raise ValueError('Overridable initializers are unsupported')
    for name,array in constants.items(): info[name] = {'shape':list(array.shape),'dtype':helper.np_dtype_to_tensor_dtype(array.dtype)}
    nodes, folded = [], Counter()
    def spec(name):
        if name not in info: raise ValueError(f'Unknown or dynamic shape: {name}')
        return info[name]
    for node in model.graph.node:
        if node.domain: raise ValueError(f'Custom domain: {node.domain}')
        if node.op_type == 'Shape' and node.input[0] in info:
            shape = info[node.input[0]]['shape']; attrs = {a.name:helper.get_attribute_value(a) for a in node.attribute}
            result = [np.array(shape[slice(attrs.get('start'),attrs.get('end'))],dtype=np.int64)]
        elif all(name in constants for name in node.input):
            result = ReferenceEvaluator(node).run(None,{name:constants[name] for name in node.input})
        else: result = None
        if result is not None:
            for name,array in zip(node.output,result,strict=True):
                constants[name] = np.asarray(array)
                info[name] = {'shape':list(array.shape),'dtype':helper.np_dtype_to_tensor_dtype(array.dtype)}
            folded[node.op_type] += 1
            continue
        op = node.op_type
        if op not in SUPPORTED: raise ValueError(f'Unsupported operation: {op} ({node.name})')
        if len(node.output)!=1 or any(not name for name in node.input): raise ValueError('Optional inputs/extra outputs unsupported')
        attrs = {a.name:helper.get_attribute_value(a) for a in node.attribute}
        if set(attrs)-ATTRS.get(op,set()): raise ValueError(f'Unsupported attributes: {op} {attrs}')
        inputs = list(node.input)
        shape = spec(inputs[0])['shape']
        if op == 'Slice':
            if not 3<=len(inputs)<=5 or any(n not in constants or constants[n].dtype!=np.int64 or constants[n].ndim!=1 for n in inputs[1:]):
                raise ValueError('Slice requires constant one-dimensional i64 controls')
            starts,ends=[constants[n].tolist() for n in inputs[1:3]]
            axes=constants[inputs[3]].tolist() if len(inputs)>=4 else list(range(len(starts)))
            steps=constants[inputs[4]].tolist() if len(inputs)==5 else [1]*len(starts)
            if not len(starts)==len(ends)==len(axes)==len(steps) or any(s!=1 for s in steps):raise ValueError('Slice supports unit positive steps only')
            if any(a < -len(shape) or a >= len(shape) for a in axes):raise ValueError('Slice axis out of range')
            axes=[a%len(shape) for a in axes]
            if len(set(axes))!=len(axes):raise ValueError('Duplicate Slice axes')
            selectors=[':']*len(shape);target=list(shape)
            for start,end,axis in zip(starts,ends,axes):
                start,end,_=slice(start,end).indices(shape[axis]);target[axis]=max(0,end-start)
                selectors[axis]=f'{start}:{end}'
            if min(target)<=0:raise ValueError('Empty Slice unsupported')
            attrs={'selectors':selectors,'shape':target};inputs=inputs[:1]
        if op == 'Unsqueeze':
            axes = constants.get(inputs[1]) if len(inputs)==2 else None
            rank = len(shape) + (axes.size if axes is not None else 0)
            if axes is None or axes.dtype!=np.int64 or axes.ndim!=1 or any(a < -rank or a >= rank for a in axes):
                raise ValueError('Unsqueeze requires constant one-dimensional i64 axes in range')
            axes = [int(a) % rank for a in axes]
            if len(set(axes))!=len(axes): raise ValueError('Duplicate Unsqueeze axes')
            target = iter(shape)
            attrs = {'shape':[1 if i in axes else next(target) for i in range(rank)]}
            inputs = inputs[:1]
            op = 'Reshape'
        elif op == 'Reshape':
            if attrs.get('allowzero',0)!=0 or inputs[1] not in constants: raise ValueError('Reshape requires constant shape, allowzero=0')
            target = constants[inputs[1]].tolist()
            target = [shape[i] if d==0 else d for i,d in enumerate(target)]
            attrs = {'shape':list(np.empty(shape,dtype=np.int8).reshape(target).shape)}
            inputs = inputs[:1]
        if op == 'Gather':
            index = constants.get(inputs[1])
            if index is None or index.ndim!=0 or index.dtype!=np.int64: raise ValueError('Only constant scalar i64 Gather indices supported')
            attrs['index'] = int(index); inputs = inputs[:1]
        if op in {'Gather','Concat','Softmax','LayerNormalization'}:
            axis = attrs.get('axis',-1 if op in {'Softmax','LayerNormalization'} else 0)
            if not -len(shape)<=axis<len(shape): raise ValueError('Axis out of range')
            attrs['axis'] = axis % len(shape)
        if op == 'LayerNormalization':
            if attrs['axis']!=len(shape)-1 or attrs.get('stash_type',1)!=1 or len(inputs)!=3: raise ValueError('Only last-axis f32 affine LayerNormalization supported')
        if op == 'Cast' and attrs['to']!=1: raise ValueError('Only f32 identity Cast supported')
        if op == 'Clip':
            if len(inputs)!=3 or any(n not in constants or constants[n].ndim!=0 for n in inputs[1:]):
                raise ValueError('Clip requires constant scalar min/max')
            lo,hi = [float(constants[n]) for n in inputs[1:]]
            if not np.isfinite([lo,hi]).all() or lo>hi: raise ValueError('Invalid Clip bounds')
            attrs={'min':lo,'max':hi};inputs=inputs[:1]
        if op == 'Pad':
            if len(shape)!=4 or len(inputs) not in [2,3] or attrs.get('mode',b'constant')!=b'constant':
                raise ValueError('Only NCHW constant-zero spatial Pad supported')
            pads=constants.get(inputs[1]);fill=constants.get(inputs[2]) if len(inputs)==3 else np.array(0)
            if pads is None or pads.dtype!=np.int64 or pads.shape!=(8,) or fill is None or fill.ndim!=0 or float(fill)!=0:
                raise ValueError('Pad requires constant i64 pads and zero fill')
            pads=pads.tolist()
            if any(v<0 for v in pads) or any(pads[i]!=0 for i in [0,1,4,5]):raise ValueError('Only nonnegative spatial padding supported')
            attrs={'pads':[pads[2],pads[3],pads[6],pads[7]]};inputs=inputs[:1]
        if op == 'Flatten':
            axis=attrs.get('axis',1)
            if not -len(shape)<=axis<=len(shape):raise ValueError('Invalid Flatten axis')
            if axis<0:axis+=len(shape)
            attrs={'shape':[int(np.prod(shape[:axis])),int(np.prod(shape[axis:]))]}
        if op == 'GlobalAveragePool' and len(shape)!=4:raise ValueError('Only NCHW GlobalAveragePool supported')
        if op == 'Conv':
            w = spec(inputs[1])['shape']
            if len(shape) not in [3,4] or len(w)!=len(shape) or len(inputs) not in [2,3] or inputs[1] not in constants or (len(inputs)==3 and inputs[2] not in constants): raise ValueError('Conv requires NCW/NCHW and constant weights/optional bias')
            spatial_rank=len(shape)-2
            kernel=attrs.get('kernel_shape',w[2:]);strides=attrs.get('strides',[1]*spatial_rank)
            dilations=attrs.get('dilations',[1]*spatial_rank);pads=attrs.get('pads',[0]*(2*spatial_rank));group=attrs.get('group',1)
            if (attrs.get('auto_pad',b'NOTSET')!=b'NOTSET' or kernel!=w[2:] or
                len(strides)!=spatial_rank or len(dilations)!=spatial_rank or len(pads)!=2*spatial_rank or min(strides+dilations)<=0 or min(pads)<0 or
                group<=0 or shape[1]!=w[1]*group or w[0]%group or
                (len(inputs)==3 and spec(inputs[2])['shape']!=[w[0]])):
                raise ValueError('Invalid Conv geometry or unsupported auto_pad')
            attrs={'kernel':kernel,'strides':strides,'dilations':dilations,'pads':pads,'group':group}
        # Propagate concrete shapes after constant folding; ONNX's first pass
        # cannot resolve downstream dimensions through the exported Expand chain.
        out_shape = list(shape)
        if op in {'Add','Mul','Div'}: out_shape = list(np.broadcast_shapes(*[spec(n)['shape'] for n in inputs]))
        elif op in {'Reshape','Flatten','Slice'}: out_shape = attrs['shape']
        elif op == 'GlobalAveragePool': out_shape = shape[:2]+[1,1]
        elif op == 'Pad': out_shape = shape[:2]+[shape[2]+attrs['pads'][0]+attrs['pads'][2],shape[3]+attrs['pads'][1]+attrs['pads'][3]]
        elif op == 'Transpose': out_shape = [shape[i] for i in attrs.get('perm',list(reversed(range(len(shape)))))]
        elif op == 'Gather': out_shape.pop(attrs['axis'])
        elif op == 'Concat':
            axis = attrs['axis']; out_shape[axis] = sum(spec(n)['shape'][axis] for n in inputs)
        elif op == 'Conv':
            out_shape = [shape[0],spec(inputs[1])['shape'][0]]+[(shape[i+2]+attrs['pads'][i]+attrs['pads'][i+len(shape)-2]-attrs['dilations'][i]*(attrs['kernel'][i]-1)-1)//attrs['strides'][i]+1 for i in range(len(shape)-2)]
            if min(out_shape)<=0:raise ValueError('Empty Conv output unsupported')
        elif op == 'MatMul':
            right = spec(inputs[1])['shape']
            if len(shape)<2 or len(right)<2 or shape[-1]!=right[-2]: raise ValueError('Only rank >=2 compatible MatMul supported')
            out_shape = [*np.broadcast_shapes(shape[:-2],right[:-2]),shape[-2],right[-1]]
        elif op == 'Gemm':
            right = spec(inputs[1])['shape']
            if len(shape)!=2 or len(right)!=2: raise ValueError('Gemm requires matrices')
            out_shape = [shape[1] if attrs.get('transA',0) else shape[0], right[0] if attrs.get('transB',0) else right[1]]
        expected = info.get(node.output[0])
        if expected and (expected['shape']!=out_shape or expected['dtype']!=1): raise ValueError(f'Inconsistent inferred output: {node.name}')
        info[node.output[0]] = {'shape':out_shape,'dtype':1}
        if any(spec(name)['dtype']!=1 for name in inputs+[node.output[0]]): raise ValueError(f'Runtime tensors must be f32: {node.name}')
        nodes.append({'op':op,'name':node.name,'inputs':inputs,'output':node.output[0], 'shape':spec(node.output[0])['shape'], 'attrs':attrs})
    inputs = {v.name:spec(v.name)['shape'] for v in model.graph.input if v.name not in constants}
    if not inputs or any(spec(n)['dtype']!=1 for n in inputs): raise ValueError('Static f32 inputs required')
    outputs = [v.name for v in model.graph.output]
    if any(spec(n)['dtype']!=1 for n in outputs): raise ValueError('Only f32 outputs supported')
    used = {name for node in nodes for name in node['inputs']} | set(outputs)
    weights = {name:np.array(value,dtype=np.float32,copy=True,order='C') for name,value in constants.items() if name in used}
    output.mkdir(parents=True,exist_ok=True)
    save_file(weights,str(output/'weights.safetensors'))
    manifest = {'format':'affon-onnx-static/v1','opset':17,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'inputs':inputs,'outputs':outputs,'constants':{n:list(v.shape) for n,v in weights.items()},'nodes':nodes,
        'conversion':{'original_nodes':len(model.graph.node),'folded_operators':dict(folded),'runtime_operators':dict(Counter(n['op'] for n in nodes))}}
    (output/'graph.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();m=convert(a.source,a.output);print(json.dumps(m['conversion'],indent=2))
