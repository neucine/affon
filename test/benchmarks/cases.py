"""Curated f32 cases. Explicit cases are coverage, not exhaustive support claims."""

def cases():
    result = []
    for op in ['abs', 'exp', 'log', 'neg', 'sqrt', 'sign', 'relu', 'sigmoid', 'silu', 'tanh', 'erf', 'gelu']:
        for shape in [[17], [64, 127]]:
            result.append(dict(id=f'{op}-'+'x'.join(map(str, shape)), op=op, a=shape, family='contiguous'))
    for op in ['add', 'sub', 'mul', 'div']:
        for family, a, b in [('tiny', [17], [17]), ('broadcast', [32, 127], [127]), ('transposed', [127, 32], [32, 127])]:
            result.append(dict(id=f'{op}-{family}', op=op, a=a, b=b, family=family, transpose_a=family=='transposed'))
    for op in ['softmax', 'layer_norm', 'sum_axis', 'mean_axis']:
        for axis in [0, 1]:
            result.append(dict(id=f'{op}-axis{axis}', op=op, a=[16, 127], axis=axis, family='axis'))
    for op in ['sum_all', 'mean_all']:
        result.append(dict(id=op, op=op, a=[32, 127], family='contiguous'))
    for op in ['min', 'max', 'variance', 'std', 'argmin', 'argmax']:
        for layout in ['contiguous', 'transposed']:
            for axis in [None, 0, 1]:
                result.append(dict(id=f'{op}-{layout}-{axis}', op=op+('_all' if axis is None else '_axis'),
                                   a=[16,127], axis=axis, transpose_a=layout=='transposed', family='reduction'))
    for op in ['sum','mean','min','max','variance','std']:
        for variant in ['offset','large-transposed']:
            result.append(dict(id=f'whole-{op}-{variant}',op=op+'_all',a=[18,129] if variant=='offset' else [128,127],
                               axis=None,narrow_inputs=variant=='offset',transpose_a=variant=='large-transposed',family='reduction'))
    for op in ['sum','mean']:
        result.append(dict(id=f'whole-{op}-transposed',op=op+'_all',a=[16,127],transpose_a=True,family='reduction'))
    # Serial/parallel dispatch boundary and an odd larger dense input.
    for op in ['sum','mean','min','max','variance','std']:
        for size in [17,255,256,257,65537]:
            result.append(dict(id=f'whole-{op}-dense-{size}',op=op+'_all',a=[size],
                               family='reduction-boundary'))
    # Size sweep across dense, transposed and interior-offset whole reductions.
    for op in ['sum','mean','min','max','variance','std']:
        for side in [256,1024,2048]:
            for layout in ['dense','transposed','offset']:
                shape = [side+2,side+1] if layout=='offset' else [side,side-1]
                result.append(dict(id=f'scale-{op}-{side}-{layout}',op=op+'_all',a=shape,
                                   transpose_a=layout=='transposed',narrow_inputs=layout=='offset',
                                   family='reduction-scaling'))
    for op in ['sum','mean','min','max','variance','std']:
        for variant,shape,axis,transpose in [
            ('long-inner',[64,4097],1,False),('long-outer',[4097,64],0,False),
            ('nd-inner',[2,4,4097],2,False),('nd-middle',[2,4097,4],1,False),
            ('transposed',[64,4097],0,True)]:
            result.append(dict(id=f'axis-scale-{op}-{variant}',op=op+'_axis',a=shape,axis=axis,
                               transpose_a=transpose,family='axis-reduction-scaling'))
    for op in ['gather', 'index_select']:
        for axis in [0, 1]:
            for layout in ['contiguous', 'transposed']:
                shape = [16,127] if layout=='contiguous' else [127,16]
                index = list(shape) if op=='gather' else [7]
                if op=='gather': index[axis]=7
                result.append(dict(id=f'{op}-{axis}-{layout}', op=op, a=[16,127], axis=axis,
                                   index=index, index_bound=shape[axis], transpose_a=layout=='transposed', family='indexing'))
    for layout in ['contiguous', 'transposed']:
        for op in ['reshape', 'contiguous', 'permute', 'slice', 'squeeze', 'unsqueeze', 'cat', 'stack']:
            result.append(dict(id=f'{op}-{layout}', op=op, a=[16,127], transpose_a=layout=='transposed',
                               target=[2032], axes=[1,0], axis=0, family='layout'))
    for op, axes in [('cat',[1]),('stack',[1,2])]:
        for axis in axes:
            for layout in ['contiguous','transposed']:
                result.append(dict(id=f'{op}-axis{axis}-{layout}',op=op,a=[4,7],b=[4,7],axis=axis,
                                   transpose_a=layout=='transposed',transpose_b=layout=='transposed',family='layout'))
    for op,axis in [('cat',1),('stack',2)]:
        for transposed in [False,True]:
            result.append(dict(id=f'{op}-offset'+('-transposed' if transposed else ''),op=op,a=[6,9],b=[6,9],axis=axis,
                               narrow_inputs=True,transpose_a=transposed,transpose_b=transposed,family='layout'))
    # Nontrivial singleton removal, in addition to squeeze's no-op cases above.
    result.append(dict(id='squeeze-singletons', op='squeeze', a=[1,16,1,127], family='layout'))
    # Either side of the CURRENT layout-aware MPS dispatch thresholds, plus
    # transposed/batched and rank-two equivalents. Not a forced-kernel benchmark.
    for m, k, n in [(2, 7, 3), (1, 255, 256), (1, 256, 256), (63, 128, 128), (64, 128, 128)]:
        for rank in [2, 3]:
            result.append(dict(id=f'matmul-{m}-{k}-{n}-rank{rank}', op='matmul', a=([1] if rank==3 else [])+[m,k], b=[k,n], family='dispatch-boundary'))
    result.append(dict(id='matmul-batched-transpose', op='matmul', a=[2,16,32], b=[2,64,32], transpose_b=True, family='batched-transpose'))
    for length in [4, 20, 35]:
        for mode in ['eager', 'scoped']:
            result.append(dict(id=f'chain-{length}-{mode}', kind='sequence', op='add', operations=['relu','add'], a=[32,127], b=[127], length=length, mode=mode, family='dependent-chain'))
    for m, width in [(2, 16), (64, 128)]:
        for mode in ['eager', 'scoped']:
            result.append(dict(id=f'matmul-chain-{m}-{width}-{mode}', kind='sequence', op='matmul',
                               operations=['matmul','relu'], a=[m,width], b=[width,width],
                               length=20, mode=mode, family='dependent-matmul-chain'))
    for fused in ['bias', 'gelu']:
        for mode in ['eager', 'scoped']:
            result.append(dict(id=f'fused-matmul-{fused}-{mode}', kind='sequence', op='matmul',
                               fused=fused, operations=['matmul','add']+(['gelu'] if fused=='gelu' else []),
                               a=[2,16], b=[16,16], length=40, mode=mode, family='compiled-epilogue-chain'))
    for c in result:
        if c['family']=='layout':
            for key,ops in [('target',['reshape']),('axes',['permute']),('axis',['unsqueeze','cat','stack'])]:
                if c['op'] not in ops: c.pop(key,None)
        c['completion'] = 'host-view' if c['op'] in ['reshape','permute','squeeze','unsqueeze'] else 'device-complete'
        c['torch_completion'] = 'host-view' if c['op']=='slice' else c['completion']
        if c['op'] in ['argmin_all','argmax_all'] and c.get('transpose_a'):
            c['unsupported_devices'] = {'mps':'PyTorch 2.9 MPS all-element arg reduction rejects this transposed view'}
        if c['op']=='reshape' and c.get('transpose_a'):
            c['unsupported'] = 'Affon eager reshape requires contiguous input; PyTorch reshape would copy'
    return [dict(kind='operator', mode='eager', **c) if 'kind' not in c else c for c in result]
