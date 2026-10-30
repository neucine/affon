"""Deterministic reduce-to-shape values and independent reduction reference."""
import math


def values(case):
    rows, cols = case['shape']
    source = [[(row * 3 + col) / 8 for col in range(cols)] for row in range(rows)]
    if '/offset_' in case['id']:
        return [[0.0] * (cols + 1), *[[0.0, *row] for row in source]]
    if case['id'].split('/')[3] == 'strided':
        return [[item for value in row for item in (value, 0.0)] for row in source]
    return source


def reference(case):
    rows, cols = case['shape']
    target = case['target_shape']
    if not target:
        return [math.fsum((row * 3 + col) / 8 for row in range(rows) for col in range(cols))]
    buckets = [[] for _ in range(math.prod(target))]
    for row in range(rows):
        for col in range(cols):
            if len(target) == 1:
                index = col if target[0] == cols else 0
            else:
                index = (row if target[0] == rows else 0) * target[1] + (col if target[1] == cols else 0)
            buckets[index].append((row * 3 + col) / 8)
    return [math.fsum(bucket) for bucket in buckets]
