#!/usr/bin/env python3
"""Run declared Affon capability probes; retain failures as unknown capability."""
import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test/parity/fixtures'))
from quantized_fixture import reference as quantized_reference
from scatter_fixture import reference as scatter_reference
from dense_loss_fixture import reference as dense_loss_reference
from reduce_shape_fixture import reference as reduce_shape_reference
from topk_fixture import reference as topk_reference
from topk_fixture import output_hash as topk_output_hash
from index_select_fixture import reference as batched_index_select_reference


def has_layout(case, layout):
    if layout == 'strided' and '_offset' in case['id']:
        return False
    return f'/{layout}' in case['id'] or f'_{layout}' in case['id']


def is_outer_axis(case):
    return '/outer_axis' in case['id']


def validate_catalog(catalog, known_ids=None):
    if catalog.get("schema_version") != 1 or catalog.get("suite") != "affon_capability_probes":
        raise ValueError("invalid capability probe catalog")
    if known_ids is None:
        known_ids = {case.get('id') for case in catalog.get('cases', [])}
    seen = set()
    for case in catalog.get("cases", []):
        case_id = case.get("id")
        if case_id not in known_ids or case_id in seen:
            raise ValueError(f"unknown or duplicate probe cell: {case_id}")
        seen.add(case_id)
        backend, dtype, op, *_ = case_id.split("/")
        if (dtype, op) != (case.get("dtype"), case.get("op")) or backend not in {"cpu", "metal", "cuda"}:
            raise ValueError(f"{case_id}: probe contract mismatch")
        shape = case.get("shape")
        if not isinstance(shape, list) or (not shape and not case_id.endswith('/scalar')) or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"{case_id}: invalid shape")
        if op == 'one_hot' and (dtype != 'f32' or len(shape) not in (2, 3) or
                                type(case.get('classes')) is not int or case['classes'] < 2):
            raise ValueError(f"{case_id}: one_hot requires 2D or 3D i64 indices, f32 output, and a positive class count")
        if op == 'quantized_matmul':
            if len(shape) != 3 or case.get('bits') not in (2, 4, 8) or type(case.get('group_size')) is not int or case['group_size'] < 1 or shape[1] % (32 // case['bits']) or shape[1] % case['group_size']:
                raise ValueError(f'{case_id}: invalid quantized matmul packing or group contract')
        if op in {'embedding', 'index_select', 'gather', 'cross_entropy_indexed'}:
            indices = case.get('index_shape')
            axis = case.get('index_axis', 1 if op == 'gather' else 0)
            if type(axis) is not int or axis not in range(len(shape)):
                raise ValueError(f"{case_id}: invalid index axis")
            if not isinstance(indices, list) or len(indices) not in (1, 2) or any(type(n) is not int or n < 1 for n in indices):
                raise ValueError(f"{case_id}: invalid index shape")
            if (len(shape) != 2 and not (op == 'index_select' and len(shape) == 3 and '/batched' in case_id)
                    or (op == 'gather' and (len(indices) != 2 or indices[1 - axis] != shape[1 - axis]))
                    or (op in {'index_select', 'cross_entropy_indexed'} and len(indices) != 1)
                    or (op == 'cross_entropy_indexed' and indices != [shape[0]])):
                raise ValueError(f"{case_id}: incompatible index shape")
        if op == 'scatter_add':
            indices = case.get('index_shape')
            axis = case.get('index_axis')
            if (len(shape) not in (2, 3) or type(axis) is not int or axis not in range(len(shape)) or
                    not isinstance(indices, list) or len(indices) != len(shape) or
                    any(type(n) is not int or n < 1 for n in indices) or
                    any(indices[dimension] != size for dimension, size in enumerate(shape) if dimension != axis) or
                    (len(shape) == 3 and '/batched' not in case_id)):
                raise ValueError(f'{case_id}: incompatible scatter index and update shapes')
        if op == 'topk' and (dtype not in {'f32', 'bf16'} or len(shape) not in (2, 3) or
                             case.get('axis') not in range(len(shape)) or type(case.get('k')) is not int or
                             not 1 <= case['k'] <= shape[case['axis']] or
                             type(case.get('largest', True)) is not bool or
                             (len(shape) == 3 and '/batched' not in case_id) or
                             (len(case_id.split('/')) > 4 and case_id.split('/')[4].split('_')[0] not in {'dense', 'strided', 'offset'})):
            raise ValueError(f'{case_id}: invalid topk contract')
        if op in {'cross_entropy', 'log_softmax_nll'} and (len(shape) not in (2, 3) or shape[1] < 3 or
                (len(shape) == 3 and '/batched' not in case_id)):
            raise ValueError(f'{case_id}: dense cross entropy requires rows and at least three classes')
        if op == 'reduce_to_shape':
            target = case.get('target_shape')
            if dtype != 'f32' or len(shape) != 2 or not isinstance(target, list) or len(target) not in (0, 1, 2) or (len(target) == 0 and not case_id.endswith('/scalar')) or any(type(n) is not int or n < 1 or (n != 1 and n != shape[len(shape) - len(target) + index]) for index, n in enumerate(target)):
                raise ValueError(f'{case_id}: incompatible f32 reduction target shape')
    return catalog["cases"]


def expected(case):
    shape, op = case["shape"], case["op"]
    if op == 'index_select' and len(shape) == 3:
        return batched_index_select_reference(case)
    if op == 'quantized_matmul':
        return quantized_reference(case)
    if op == 'scatter_add':
        return scatter_reference(case)
    if op == 'topk':
        return topk_reference(case)
    if op in {'cross_entropy', 'log_softmax_nll'}:
        return dense_loss_reference(case)
    if op == 'reduce_to_shape':
        return reduce_shape_reference(case)
    broadcasting = '/broadcast' in case['id']
    invalid = '/invalid_index' in case['id']
    def bf16(number):
        bits = struct.unpack('<I', struct.pack('<f', number))[0]
        bits += 0x7fff + ((bits >> 16) & 1)
        return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]
    def rounded(number):
        return bf16(number) if case['dtype'] == 'bf16' else number
    def value(indices, seed):
        return rounded(seed + sum(index / 8 for index in indices))
    def viewed_value(row, col, seed):
        if has_layout(case, 'strided'):
            col *= 2
        elif has_layout(case, 'offset'):
            row, col = row + 1, col + 1
        if seed == 0.5 and (has_layout(case, 'strided') or has_layout(case, 'offset')):
            return rounded(seed + row / 4 + col / 16)
        return value((row, col), seed)
    if not shape:
        x, y = rounded(1), rounded(0.5)
        scalar = {'add': lambda: x + y, 'sub': lambda: x - y,
                  'mul': lambda: x * y, 'div': lambda: x / y,
                  'eq': lambda: int(x == 1.5), 'lt': lambda: int(x < 1.5),
                  'gt': lambda: int(x > 1.5),
                  'neg': lambda: -x, 'abs': lambda: abs(x),
                  'exp': lambda: math.exp(x), 'log': lambda: math.log(x),
                  'sqrt': lambda: math.sqrt(x), 'sign': lambda: (x > 0) - (x < 0),
                  'relu': lambda: max(x, 0), 'sigmoid': lambda: 1 / (1 + math.exp(-x)),
                  'silu': lambda: x / (1 + math.exp(-x)), 'tanh': lambda: math.tanh(x),
                  'sin': lambda: math.sin(x), 'cos': lambda: math.cos(x),
                  'erf': lambda: math.erf(x),
                  'gelu': lambda: x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))) / 2,
                  'clamp': lambda: min(2.25, max(1.25, x)),
                  'where': lambda: x, 'masked_fill': lambda: -2,
                  'cast': lambda: bf16(1.1),
                  'sum_all': lambda: x, 'mean_all': lambda: x,
                  'min_all': lambda: x, 'max_all': lambda: x,
                  'variance_all': lambda: 0, 'std_all': lambda: 0}
        return [rounded(scalar[op]())]
    if op in {'embedding', 'index_select', 'gather'}:
        indices = case['index_shape']
        axis = case.get('index_axis', 1 if op == 'gather' else 0)
        # The invalid_index regime feeds indices past the last valid position; both
        # PyTorch and Affon zero-fill those selections, so the reference mirrors that.
        if invalid:
            def index_value(row, col):
                # Mirrors the probe worker: even slots stay in range, odd slots overshoot.
                return ((col * 3 + 1) % shape[axis]) if (row + col) % 2 == 0 else shape[axis] + col
            if op == 'gather':
                return [0.0 if (chosen := index_value(row, col)) >= shape[axis] else
                        viewed_value(*(chosen, col) if axis == 0 else (row, chosen), 1)
                        for row in range(indices[0]) for col in range(indices[1])]
            if axis == 0:
                return [0.0 if (selected := index_value(0, col)) >= shape[0] else viewed_value(selected, col, 1)
                        for col in range(indices[0]) for _ in range(shape[1])]
            return [0.0 if (selected := index_value(0, col)) >= shape[1] else viewed_value(row, selected, 1)
                    for row in range(shape[0]) for col in range(indices[0])]
        if op == 'gather':
            return [viewed_value((col * 3 + row) % shape[0], col, 1) if axis == 0 else viewed_value(row, (col * 3 + row) % shape[1], 1)
                    for row in range(indices[0]) for col in range(indices[1])]
        if len(indices) == 1:
            selected = [(col * 3 + 1) % shape[axis] for col in range(indices[0])]
        else:
            selected = [(row * 5 + col * 3 + 1) % shape[0]
                        for row in range(indices[0]) for col in range(indices[1])]
        if op == 'embedding' or axis == 0:
            return [viewed_value(row, col, 1) for row in selected for col in range(shape[1])]
        return [viewed_value(row, col, 1) for row in range(shape[0]) for col in selected]
    if op == "matmul":
        if len(shape) == 4:
            batches, m, k, n = shape
            return [sum((1 + row / 8 + index / 8) *
                        (0.5 + batch / 8 + index / 8 + col / 8)
                        for index in range(k))
                    for batch in range(batches) for row in range(m) for col in range(n)]
        m, k, n = shape
        lhs_strided = '/strided/lhs_' in case['id']
        rhs_strided = has_layout(case, 'strided') and not lhs_strided
        lhs_step = 1 / 4 if lhs_strided else 1 / 8
        rhs_step = 1 / 8
        index_sum = k * (k - 1) / 2
        index_square_sum = k * (k - 1) * (2 * k - 1) / 6
        return [k * (1 + row / 8) * (0.5 + (col / 4 if rhs_strided else col / 8)) +
                ((1 + row / 8) * rhs_step + (0.5 + (col / 4 if rhs_strided else col / 8)) * lhs_step) * index_sum +
                lhs_step * rhs_step * index_square_sum
                for row in range(m) for col in range(n)]
    if op == 'dot':
        def physical(index):
            return index * 2 if has_layout(case, 'strided') else index + 1 if has_layout(case, 'offset') else index
        return [rounded(math.fsum(value((physical(index),), 1) * value((physical(index),), 0.5)
                                  for index in range(shape[0])))]
    if op == 'one_hot' and len(shape) == 3:
        return [float(class_index == (batch * 5 + row * 3 + col) % case['classes'])
                for batch in range(shape[0]) for row in range(shape[1]) for col in range(shape[2])
                for class_index in range(case['classes'])]
    rows, cols = shape
    if op == 'one_hot':
        def index_value(row, col):
            physical_row = row + 1 if has_layout(case, 'offset') else row
            physical_col = col * 2 if has_layout(case, 'strided') else col + 1 if has_layout(case, 'offset') else col
            if invalid and (physical_row + physical_col) % 3 != 0:
                return case['classes'] + physical_col if (physical_row + physical_col) % 2 else -1
            return (physical_row * 3 + physical_col) % case['classes']
        return [float(class_index == index_value(row, col))
                for row in range(rows) for col in range(cols) for class_index in range(case['classes'])]
    if op == 'cast':
        def physical(row, col):
            if has_layout(case, 'strided'):
                col *= 2
            elif has_layout(case, 'offset'):
                row, col = row + 1, col + 1
            return row, col
        return [bf16(1.1 + physical(row, col)[0] / 7 + physical(row, col)[1] / 13)
                for row in range(rows) for col in range(cols)]
    if op in {'softmax', 'log_softmax', 'layer_norm', 'rms_norm'} and is_outer_axis(case):
        source = [[viewed_value(row, col, 1) for col in range(cols)] for row in range(rows)]
        if op in {'layer_norm', 'rms_norm'}:
            items = [item for row in source for item in row]
            average = math.fsum(items) / len(items)
            variance = math.fsum((item - average) ** 2 for item in items) / len(items)
            return [rounded((item - average) / math.sqrt(variance + 1e-5) if op == 'layer_norm' else item / math.sqrt(math.fsum(x*x for x in items) / len(items) + 1e-5)) for item in items]
        result = [[0.0] * cols for _ in range(rows)]
        for col in range(cols):
            items = [source[row][col] for row in range(rows)]
            peak = max(items)
            weights = [math.exp(item - peak) for item in items]
            for row, item in enumerate(weights):
                result[row][col] = rounded(math.log(item / math.fsum(weights)) if op == 'log_softmax' else item / math.fsum(weights))
        return [item for row in result for item in row]
    if op in {'eq', 'lt', 'gt'}:
        def comparison_right(row, col):
            if broadcasting:
                row = 0
            elif has_layout(case, 'strided'):
                col *= 2
            elif has_layout(case, 'offset'):
                row, col = row + 1, col + 1
            return rounded(1 + row / 8 + col / 8 + (((row + col) % 3) - 1) / 2)
        compare = {'eq': lambda a, b: a == b, 'lt': lambda a, b: a < b,
                   'gt': lambda a, b: a > b}[op]
        return [int(compare(viewed_value(row, col, 1), comparison_right(row, col)))
                for row in range(rows) for col in range(cols)]
    if op in {'where', 'masked_fill'}:
        def mask(row, col):
            return (0 if op == 'masked_fill' and broadcasting else row) + col & 1
        return [rounded((viewed_value(row, col, 1) if mask(row, col) else
                         viewed_value(0 if broadcasting else row, col, 0.5))
                        if op == 'where' else
                        (-2 if mask(row, col) else viewed_value(row, col, 1)))
                for row in range(rows) for col in range(cols)]
    if op == 'cross_entropy_indexed':
        losses = []
        for row in range(rows):
            logits = [viewed_value(row, col, 1) for col in range(cols)]
            peak = max(logits)
            logsumexp = peak + math.log(math.fsum(math.exp(item - peak) for item in logits))
            losses.append(logsumexp - logits[(row * 3 + 1) % cols])
        return [rounded(math.fsum(losses) / rows)]
    if op in {'argmin_all', 'argmax_all', 'argmin_axis', 'argmax_axis'}:
        lower = op.startswith('argmin')
        def selected(items):
            return min(range(len(items)), key=lambda index: items[index]) if lower else max(range(len(items)), key=lambda index: items[index])
        if op.endswith('_all'):
            return [selected([viewed_value(row, col, 1) for row in range(rows) for col in range(cols)])]
        outer = is_outer_axis(case)
        return [selected([viewed_value(row, col, 1) for row, col in
                          ([(row, index) for row in range(rows)] if outer else [(index, col) for col in range(cols)])])
                for index in range(cols if outer else rows)]
    if op in {'sum_all', 'mean_all', 'min_all', 'max_all', 'variance_all', 'std_all'}:
        items = [viewed_value(row, col, 1) for row in range(rows) for col in range(cols)]
        average = math.fsum(items) / len(items)
        population_variance = math.fsum((item - average) ** 2 for item in items) / len(items)
        statistic = {'sum_all': lambda: math.fsum(items), 'mean_all': lambda: average,
                     'min_all': lambda: min(items), 'max_all': lambda: max(items),
                     'variance_all': lambda: population_variance,
                     'std_all': lambda: math.sqrt(population_variance)}[op]()
        return [rounded(statistic)]
    if op in {'reshape', 'transpose', 'permute', 'slice', 'contiguous', 'cat', 'stack', 'squeeze', 'unsqueeze'}:
        left = [[viewed_value(row, col, 1) for col in range(cols)] for row in range(rows)]
        right = [[viewed_value(row, col, 0.5) for col in range(cols)] for row in range(rows)]
        if op == 'reshape' and case['id'].endswith('/materialized'):
            return [left[row][col] for col in range(cols) for row in range(rows)]
        if op == 'contiguous' and case['id'].endswith('/view_only'):
            return [item for row in left for item in row]
        if op in {'reshape', 'stack', 'squeeze', 'unsqueeze'}:
            flat = [item for row in left for item in row]
            return flat + [item for row in right for item in row] if op == 'stack' else flat
        if op in {'transpose', 'permute', 'contiguous'}:
            return [left[row][col] for col in range(cols) for row in range(rows)]
        if op == 'slice':
            return [left[row][col] for row in range(rows // 2) for col in range(cols // 2)]
        return [item for row in range(rows) for item in left[row] + right[row]]
    if op == "add":
        return [rounded(viewed_value(row, col, 1) + viewed_value(0 if broadcasting else row, col, 0.5)) for row in range(rows) for col in range(cols)]
    if op == "mul":
        return [rounded(viewed_value(row, col, 1) * viewed_value(0 if broadcasting else row, col, 0.5)) for row in range(rows) for col in range(cols)]
    if op == "sub":
        return [rounded(viewed_value(row, col, 1) - viewed_value(0 if broadcasting else row, col, 0.5)) for row in range(rows) for col in range(cols)]
    if op == "div":
        return [rounded(viewed_value(row, col, 1) / viewed_value(0 if broadcasting else row, col, 0.5)) for row in range(rows) for col in range(cols)]
    unary = {'neg': lambda x: -x, 'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
             'relu': lambda x: max(0, x), 'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
             'abs': abs, 'sign': lambda x: (x > 0) - (x < 0),
             'silu': lambda x: x / (1 + math.exp(-x)), 'tanh': math.tanh,
             'sin': math.sin, 'cos': math.cos, 'erf': math.erf,
             'gelu': lambda x: x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))) / 2,
             'clamp': lambda x: min(2.25, max(1.25, x))}
    if op in unary:
        return [rounded(unary[op](viewed_value(row, col, 1))) for row in range(rows) for col in range(cols)]
    if op in {'layer_norm', 'rms_norm'}:
        result = []
        for row in range(rows):
            items = [viewed_value(row, col, 1) for col in range(cols)]
            mean = sum(items) / cols
            variance = sum((item - mean) ** 2 for item in items) / cols
            result.extend(rounded((item - mean) / math.sqrt(variance + 1e-5) if op == 'layer_norm' else item / math.sqrt(sum(x*x for x in items) / cols + 1e-5)) for item in items)
        return result
    if op == "sum_axis":
        outer = is_outer_axis(case)
        if outer:
            return [rounded(math.fsum(viewed_value(row, col, 1) for row in range(rows))) for col in range(cols)]
        return [rounded(sum(viewed_value(row, col, 1) for col in range(cols))) for row in range(rows)]
    if op in {'mean_axis', 'min_axis', 'max_axis', 'variance_axis', 'std_axis'}:
        result = []
        outer = is_outer_axis(case)
        for output_index in range(cols if outer else rows):
            items = ([viewed_value(row, output_index, 1) for row in range(rows)] if outer else
                     [viewed_value(output_index, col, 1) for col in range(cols)])
            average = sum(items) / len(items)
            population_variance = sum((item - average) ** 2 for item in items) / len(items)
            statistic = {'mean_axis': lambda: average, 'min_axis': lambda: min(items),
                         'max_axis': lambda: max(items), 'variance_axis': lambda: population_variance,
                         'std_axis': lambda: math.sqrt(population_variance)}[op]()
            result.append(rounded(statistic))
        return result
    if op in {"softmax", "log_softmax"}:
        result = []
        for row in range(rows):
            values = [viewed_value(row, col, 1) for col in range(cols)]
            peak = max(values)
            scaled = [math.exp(x - peak) for x in values]
            result.extend(rounded(math.log(x / sum(scaled)) if op == "log_softmax" else x / sum(scaled)) for x in scaled)
        return result
    raise ValueError(f"unsupported probe operation: {op}")


def expected_shape(case):
    if not case['shape']:
        return None
    if case['op'] == 'one_hot':
        return case['shape'] + [case['classes']]
    if case['op'] == 'index_select' and len(case['shape']) == 3:
        result = case['shape'].copy()
        result[case['index_axis']] = case['index_shape'][0]
        return result
    if case['op'] == 'quantized_matmul':
        return [case['shape'][0], case['shape'][2]]
    if case['op'] == 'scatter_add':
        return case['shape']
    if case['op'] == 'topk':
        result = case['shape'].copy()
        result[case['axis']] = case['k']
        return result
    if case['op'] in {'cross_entropy', 'log_softmax_nll'}:
        return [1]
    if case['op'] == 'reduce_to_shape':
        return case['target_shape']
    if case['op'] == 'dot':
        return []
    rows, cols = case['shape'][:2]
    if case['op'] == 'contiguous' and case['id'].endswith('/view_only') or case['op'] == 'reshape' and case['id'].endswith('/materialized'):
        return [rows, cols]
    if case['op'] == 'cross_entropy_indexed':
        return [1]
    if case['op'] in {'argmin_all', 'argmax_all'}:
        return [1]
    if case['op'] in {'argmin_axis', 'argmax_axis'}:
        return [cols if is_outer_axis(case) else rows]
    if case['op'] in {'sum_all', 'mean_all', 'min_all', 'max_all', 'variance_all', 'std_all'}:
        return [1]
    if case['op'] in {'embedding', 'index_select', 'gather'}:
        return (case['index_shape'] + [cols] if case['op'] == 'embedding' else
                [case['index_shape'][0], cols] if case['op'] == 'index_select' and case.get('index_axis', 0) == 0 else
                [rows, case['index_shape'][0]] if case['op'] == 'index_select' else case['index_shape'])
    return {
        'reshape': [cols, rows], 'transpose': [cols, rows], 'permute': [cols, rows],
        'contiguous': [cols, rows], 'slice': [rows // 2, cols // 2],
        'cat': [rows, cols * 2], 'stack': [2, rows, cols],
        'squeeze': [cols], 'unsqueeze': [1, rows, cols],
    }.get(case['op'])


def run_one(binary, case):
    env = os.environ.copy()
    env["AFFON_PROBE_ID"] = case["id"]
    env["AFFON_PROBE_SHAPE"] = json.dumps(case["shape"])
    if case['op'] == 'reduce_to_shape':
        env['AFFON_TARGET_SHAPE'] = json.dumps(case['target_shape'])
    if case['op'] == 'quantized_matmul':
        env['AFFON_Q_BITS'] = str(case['bits'])
        env['AFFON_Q_GROUP'] = str(case['group_size'])
    if 'index_shape' in case:
        env['AFFON_PROBE_INDEX_SHAPE'] = json.dumps(case['index_shape'])
    if 'index_axis' in case:
        env['AFFON_PROBE_INDEX_AXIS'] = str(case['index_axis'])
    if 'classes' in case:
        env['AFFON_PROBE_CLASSES'] = str(case['classes'])
    env["AFFON_PROBE_DENSE_INPUTS"] = "1" if case.get("dense_inputs", False) else "0"
    if case['op'] == 'topk':
        env['AFFON_TOPK_AXIS'] = str(case['axis'])
        env['AFFON_TOPK_K'] = str(case['k'])
        env['AFFON_TOPK_LARGEST'] = '1' if case.get('largest', True) else '0'
    worker = 'affon-index-select.ts' if case['op'] == 'index_select' and len(case['shape']) == 3 else 'affon-topk.ts' if case['op'] == 'topk' else 'affon-quantized.ts' if case['op'] == 'quantized_matmul' else 'affon-scatter.ts' if case['op'] == 'scatter_add' else 'affon-dense-loss.ts' if case['op'] in {'cross_entropy', 'log_softmax_nll'} else 'affon-reduce-shape.ts' if case['op'] == 'reduce_to_shape' else 'affon-probe.ts'
    process = subprocess.run([str(binary), str(ROOT / 'test/parity/workers' / worker)],
                             cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    lines = [line for line in process.stdout.splitlines() if line.startswith('{')]
    if process.returncode != 0 or len(lines) != 1:
        return {"id": case["id"], "status": "probe_failed", "returncode": process.returncode,
                "stderr": process.stderr[-4000:], "stdout": process.stdout[-1000:]}
    result = json.loads(lines[0])
    if result.get("suite") != "affon_capability_probe" or result.get("id") != case["id"] or result.get("shape") != case["shape"]:
        raise ValueError(f"{case['id']}: worker contract mismatch")
    declared_output_shape = expected_shape(case)
    expected_dtype = 'pair:' + case['dtype'] + ',i64' if case['op'] == 'topk' else 'f32' if case['op'] == 'quantized_matmul' else ('bf16' if case['dtype'] == 'f32' else 'f32') if case['op'] == 'cast' else None
    if case['op'] == 'topk' and result.get('outputs') != [
        {'shape': declared_output_shape, 'dtype': case['dtype'], 'count': math.prod(declared_output_shape)},
        {'shape': declared_output_shape, 'dtype': 'i64', 'count': math.prod(declared_output_shape)}]:
        return {'id': case['id'], 'status': 'incorrect_output_contract', 'outputs': result.get('outputs')}
    if expected_dtype and result.get('output_dtype') != expected_dtype:
        return {"id": case['id'], "status": "incorrect_dtype", "output_dtype": result.get('output_dtype'), "expected_output_dtype": expected_dtype}
    if declared_output_shape is not None and result.get('output_shape') != declared_output_shape:
        return {"id": case['id'], "status": "incorrect_shape", "shape": case['shape'],
                "output_shape": result.get('output_shape'), "expected_output_shape": declared_output_shape}
    reference = expected(case)
    samples = result.get("output_samples")
    if result.get("output_count") != len(reference) or not isinstance(samples, list) or len(samples) != min(5, len(reference)) or any(
        type(item.get("index")) is not int or item["index"] < 0 or item["index"] >= len(reference) or
        not isinstance(item.get("value"), (int, float)) or not math.isfinite(item["value"]) for item in samples
    ):
        raise ValueError(f"{case['id']}: invalid output")
    maximum_error = max(abs(item["value"] - reference[item["index"]]) for item in samples)
    topk_hash_matches = case['op'] != 'topk' or result.get('output_hash') == topk_output_hash(reference)
    relative_tolerance = 0.02 if case['dtype'] == 'bf16' else 0.00001 if case['op'] in {'variance_axis', 'std_axis'} else 0.000002
    tolerance = relative_tolerance * max(1, max(abs(x) for x in reference))
    checksum_error = abs(result.get("output_checksum", float("nan")) - sum(reference))
    checksum_tolerance = tolerance * len(reference)
    nodes = result["explanation"].get("executable", {}).get("physical_nodes", [])
    path = [{"implementation_id": node.get("implementation_id") or ("view.transpose" if node.get("view_permutation") else "view." + str(node.get("view_transform"))),
             "implementation_version": node.get("implementation_version"),
             "configuration_digest": node.get("configuration_digest"),
             **({"view_slice_ranges": node.get("view_slice_ranges"), "view_permutation": node.get("view_permutation")} if node.get("action") == "view" else {})}
            for node in nodes]
    if not path or any(not item["implementation_id"] for item in path):
        raise ValueError(f"{case['id']}: selected physical path missing")
    return {"id": case["id"], "status": "correct" if maximum_error <= tolerance and checksum_error <= checksum_tolerance and topk_hash_matches else "incorrect", "shape": case["shape"],
            "input_shapes": result.get("input_shapes"),
            "compile_options": {"dense_inputs": bool(case.get("dense_inputs", False))},
            "output_shape": result["output_shape"], "output_dtype": result.get('output_dtype'), "output_count": result["output_count"],
            **({'outputs': result['outputs']} if case['op'] == 'topk' else {}),
            "maximum_error": maximum_error, "tolerance": tolerance,
            "checksum_error": checksum_error, "checksum_tolerance": checksum_tolerance,
            **({'output_hash_matches': topk_hash_matches} if case['op'] == 'topk' else {}),
            "selected_path": path, "explanation": result["explanation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--catalog", type=Path, default=ROOT / "benchmarks/capability-probes.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda"])
    args = parser.parse_args()
    cases = validate_catalog(json.loads(args.catalog.read_text()))
    if args.backend:
        cases = [case for case in cases if case["id"].startswith(args.backend + "/")]
    if not args.binary.is_file():
        parser.error(f"binary missing: {args.binary}")
    report = {"schema_version": 1, "suite": "affon_capability_probe_run", "platform": platform.platform(),
              "zig_version": subprocess.check_output(["zig", "version"], text=True).strip(),
              "affon_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "affon_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              "cases": [run_one(args.binary.resolve(), case) for case in cases]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "correct": sum(row["status"] == "correct" for row in report["cases"]),
                      "failed": sum(row["status"] != "correct" for row in report["cases"])}))


if __name__ == "__main__":
    main()
