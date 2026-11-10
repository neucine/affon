#!/usr/bin/env python3
"""Validate stable peer selection on independent tied, signed, and strided inputs."""
import itertools
import json
import random
import torch

from stable_topk import stable_topk


def check():
    random.seed(20261105)
    count = 0
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((3, 7), (9, 5), (2, 3, 11), (3, 8, 5)):
            size = 1
            for dimension in shape:
                size *= dimension
            source = torch.tensor([random.randrange(-4, 5) for _ in range(size)], dtype=dtype).reshape(shape)
            for x in (source, source.transpose(-1, -2), source[..., ::2], torch.zeros_like(source)):
                for axis in range(x.ndim):
                    for largest in (False, True):
                        for k in sorted({1, x.shape[axis] // 2 or 1, x.shape[axis]}):
                            values, indices = stable_topk(x, k, axis, largest)
                            for coordinate in itertools.product(*(range(n) for n in values.shape)):
                                candidates = []
                                for index in range(x.shape[axis]):
                                    location = list(coordinate)
                                    location[axis] = index
                                    candidates.append((float(x[tuple(location)]), index))
                                expected = sorted(candidates, key=lambda pair: (-pair[0] if largest else pair[0], pair[1]))[coordinate[axis]]
                                actual = float(values[coordinate]), int(indices[coordinate])
                                if actual != expected:
                                    raise ValueError((dtype, shape, axis, largest, k, coordinate, actual, expected))
                            count += 1
    return {'suite': 'stable_topk_independent_cpu_validation', 'full_output_checks_passed': count,
            'dtypes': ['f32', 'bf16'], 'all_axes': True, 'largest_and_smallest': True,
            'layouts': ['dense', 'transposed', 'strided'], 'includes_all_equal_and_cutoff_ties': True}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
