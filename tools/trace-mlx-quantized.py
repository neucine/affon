#!/usr/bin/env python3
"""One compute encoder per synchronized QMM; consumed by profile-quantized-gpu.py."""
import time
import sys
import os
import numpy as np
import mlx.core as mx
for m in ([128] if os.getenv('AFFON_QMM_M128_ONLY')=='1' else [1,8,32,128]):
    n, k, bits, gs = 2048, 1024, 8, 64
    r = np.arange(n, dtype=np.uint32)[:, None]
    c = np.arange(k // 4, dtype=np.uint32)[None, :] * 4
    packed = np.zeros((n, k // 4), np.uint32)
    for lane in range(4):
        packed |= ((r * 3 + (c + lane) * 7) & 255) << (8 * lane)
    values = ((np.arange(m)[:, None] + np.arange(k)[None, :]) % 17 - 8).astype(np.float32) / 16
    x, w = mx.array(values), mx.array(packed)
    s, b = mx.full((n, k // gs), 1 / 128, mx.bfloat16), mx.full((n, k // gs), -1, mx.bfloat16)
    mx.eval(x, w, s, b)
    # Preserve the complete output during evaluation. Scalar slicing cannot
    # permit an optimizer to restrict the matrix computation in this profile.
    def graph(a):
        y = mx.quantized_matmul(a, w, s, b, bits=bits, group_size=gs)
        return y[:1, :1] if '--slice-inside' in sys.argv else y
    f = mx.compile(graph)
    actual=np.array(mx.quantized_matmul(x,w,s,b,bits=bits,group_size=gs))
    for row,col in [(0,0),(m-1,n-1)]:
        expected=sum(((row+inner)%17-8)/16*(((col*3+inner*7)&255)/128-1) for inner in range(k))
        if abs(float(actual[row,col])-expected)>1e-4:
            raise AssertionError('QMM benchmark correctness mismatch')
    warm_end=time.perf_counter()+int(os.getenv('AFFON_QMM_WARMUP_MS','300'))/1000
    while time.perf_counter()<warm_end:
        mx.eval(f(x))
    for _ in range(int(os.getenv('AFFON_QMM_SAMPLES','100'))):
        mx.eval(f(x))
    time.sleep(.15)
