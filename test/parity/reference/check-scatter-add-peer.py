#!/usr/bin/env python3
"""Check skip-invalid scatter semantics against independent coordinate updates."""
import itertools
import json
import random
import torch

from scatter_add_peer import scatter_add_skip_invalid


def check():
    random.seed(20261105)
    count = 0
    for device in ('cpu', 'cuda'):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((3, 7), (9, 5), (2, 3, 11), (3, 8, 5)):
                for axis in range(len(shape)):
                    for layout in ('dense', 'transposed', 'strided'):
                        size = 1
                        for n in shape:
                            size *= n
                        x = torch.tensor([random.randrange(-4, 5) / 8 for _ in range(size)], dtype=dtype).reshape(shape)
                        if layout == 'transposed':
                            x = x.transpose(-1, -2)
                        elif layout == 'strided':
                            x = x[..., ::2]
                        indices = torch.empty_like(x, dtype=torch.long)
                        updates = torch.empty_like(x)
                        expected = x.clone()
                        for coordinate in itertools.product(*(range(n) for n in x.shape)):
                            index = random.randrange(-2, x.shape[axis] + 3)
                            value = random.randrange(-3, 4) / 8
                            indices[coordinate] = index
                            updates[coordinate] = value
                            if 0 <= index < x.shape[axis]:
                                target = list(coordinate)
                                target[axis] = index
                                expected[tuple(target)] += value
                        actual = scatter_add_skip_invalid(x.to(device), indices.to(device), updates.to(device), axis).cpu()
                        if not torch.equal(actual, expected):
                            raise ValueError((device, dtype, shape, axis, layout))
                        count += 1
    return {'suite': 'scatter_add_skip_invalid_independent_validation', 'full_output_checks_passed': count,
            'devices': ['cpu', 'cuda'], 'dtypes': ['f32', 'bf16'], 'all_axes': True,
            'layouts': ['dense', 'transposed', 'strided'], 'negative_and_oversized_indices': True}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
