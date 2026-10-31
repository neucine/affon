#!/usr/bin/env python3
"""Matched resident Affon/PyTorch CPU and MPS cell comparisons."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test/parity/fixtures'))
from quantized_fixture import parameters as quantized_parameters, physical_parameters as quantized_physical_parameters
from scatter_fixture import physical_parameters as scatter_physical_parameters
from dense_loss_fixture import physical_parameters as dense_loss_physical_parameters
from reduce_shape_fixture import values as reduce_shape_values
from topk_fixture import physical_values as topk_physical_values
from topk_fixture import output_hash as topk_output_hash
from index_select_fixture import parameters as batched_index_select_parameters


def is_outer_axis(case):
    return '/outer_axis' in case['id']


def has_layout(case, layout):
    if layout == 'strided' and '_offset' in case['id']:
        return False
    return f'/{layout}' in case['id'] or f'_{layout}' in case['id']


def values(shape, seed):
    if not shape:
        return seed
    return [values(shape[1:], seed + index / 8) for index in range(shape[0])]


def view_right_values(shape):
    return [[0.5 + row / 4 + col / 16 for col in range(shape[1])] for row in range(shape[0])]


def compare_right_values(shape):
    if not shape:
        return 1.5
    return [[1 + row / 8 + col / 8 + (((row + col) % 3) - 1) / 2
             for col in range(shape[1])] for row in range(shape[0])]


def cast_values(shape):
    if not shape:
        return 1.1
    return [[1.1 + row / 7 + col / 13 for col in range(shape[1])]
            for row in range(shape[0])]


def inputs(case):
    shape = case["shape"]
    if case['op'] == 'index_select' and len(shape) == 3:
        return batched_index_select_parameters(case)
    if case['op'] == 'topk':
        return topk_physical_values(case), []
    if case['op'] == 'quantized_matmul':
        x, packed, _, _ = quantized_physical_parameters(case)
        return x, packed
    if case['op'] == 'scatter_add':
        base, indices, _ = scatter_physical_parameters(case)
        return base, indices
    if case['op'] in {'cross_entropy', 'log_softmax_nll'}:
        return dense_loss_physical_parameters(case)
    if case['op'] == 'reduce_to_shape':
        return reduce_shape_values(case), []
    if not shape:
        return 1.1 if case['op'] == 'cast' else 1.0, 1.5 if case['op'] in {'eq', 'lt', 'gt'} else 0.5
    if case['op'] == 'dot':
        physical = [shape[0] * 2] if has_layout(case, 'strided') else [shape[0] + 1] if has_layout(case, 'offset') else shape
        return values(physical, 1), values(physical, 0.5)
    if case['op'] == 'one_hot':
        if len(shape) == 3:
            indices = [[[(batch * 5 + row * 3 + col) % case['classes']
                         for col in range(shape[2])] for row in range(shape[1])]
                       for batch in range(shape[0])]
            return indices, values(shape, 0.5)
        physical = ([shape[0], shape[1] * 2] if has_layout(case, 'strided') else
                    [shape[0] + 1, shape[1] + 1] if has_layout(case, 'offset') else shape)
        def index_value(row, col):
            if '/invalid_index' in case['id'] and (row + col) % 3 != 0:
                return case['classes'] + col if (row + col) % 2 else -1
            return (row * 3 + col) % case['classes']
        indices = [[index_value(row, col) for col in range(physical[1])]
                   for row in range(physical[0])]
        return indices, values(physical, 0.5)
    if case['op'] in {'embedding', 'index_select', 'gather', 'cross_entropy_indexed'}:
        index_shape = case['index_shape']
        axis = case.get('index_axis', 1 if case['op'] == 'gather' else 0)
        left_shape = ([shape[0], shape[1] * 2] if has_layout(case, 'strided') else
                      [shape[0] + 1, shape[1] + 1] if has_layout(case, 'offset') else shape)
        if len(index_shape) == 1:
            indices = [(col * 3 + 1) % (shape[1] if case['op'] == 'cross_entropy_indexed' else shape[axis]) for col in range(index_shape[0])]
        else:
            indices = [[(col * 3 + row) % shape[axis] if case['op'] == 'gather' else
                        (row * 5 + col * 3 + 1) % shape[0] for col in range(index_shape[1])]
                       for row in range(index_shape[0])]
        return values(left_shape, 1), indices
    batched = case['op'] == 'matmul' and len(shape) == 4
    lhs_transposed = case['op'] == 'matmul' and '/transposed/lhs_' in case['id']
    lhs_strided = case['op'] == 'matmul' and '/strided/lhs_' in case['id']
    left = shape[1:3] if batched else [shape[1], shape[0]] if lhs_transposed else [shape[0], shape[1] * 2] if lhs_strided else shape[:2] if case["op"] == "matmul" else [shape[0], shape[1] * 2] if has_layout(case, 'strided') else [shape[0] + 1, shape[1] + 1] if has_layout(case, 'offset') else shape
    right = ([shape[0], shape[2], shape[3]] if batched else
             [shape[2], shape[1]] if '/transposed' in case['id'] and not lhs_transposed else
             [shape[1], shape[2] * 2] if has_layout(case, 'strided') and not lhs_strided else
             [shape[1], shape[2]]) if case["op"] == "matmul" else [1, shape[1]] if '/broadcast' in case['id'] else left
    return cast_values(left) if case['op'] == 'cast' else values(left, 1), compare_right_values(right) if case['op'] in {'eq', 'lt', 'gt'} else view_right_values(right) if case['op'] != 'matmul' and (has_layout(case, 'strided') or has_layout(case, 'offset')) else values(right, 0.5)


def program_graph(case):
    if case['op'] == 'topk':
        return {'id': 'topk_values_add_self_indices_v1', 'ops': ['topk', 'add_self'],
                'source': 'test/parity/workers/affon-topk.ts'}
    if case['op'] == 'quantized_matmul':
        return {"id": "quantized_matmul_add_self_v1", "ops": ["quantized_matmul", "add_self"],
                "source": "test/parity/workers/affon-quantized.ts"}
    if case["op"] == "matmul":
        return {"id": "gpt2_dense_bias_v1", "ops": ["matmul", "add_bias"],
                "source": "packages/@affon/models/src/gpt2/model.ts:dense"}
    return {"id": "sum_add_self_v1", "ops": [case["op"], "add_self"],
            "source": "synthetic_diagnostic"}


def flatten(value):
    if isinstance(value, list):
        return [item for part in value for item in flatten(part)]
    return [value]


def shape_of(value):
    return [len(value), *shape_of(value[0])] if isinstance(value, list) and value else []


def mask_values(case):
    shape = case['shape']
    if not shape:
        return 1
    rows = 1 if case['op'] == 'masked_fill' and '/broadcast' in case['id'] else shape[0]
    return [[(row + col) % 2 for col in range(shape[1])] for row in range(rows)]


def affon(binary, case, iterations, samples, kind="operator_peer", warmup_calls=20, extra_env=None):
    env = os.environ.copy()
    env.update(AFFON_CELL_ID=case["id"], AFFON_CELL_SHAPE=json.dumps(case["shape"]),
               AFFON_CELL_ITERATIONS=str(iterations), AFFON_CELL_SAMPLES=str(samples),
               AFFON_CELL_DENSE_INPUTS="1" if case.get("dense_inputs", False) else "0",
               AFFON_CELL_KIND=kind, AFFON_CELL_WARMUP=str(warmup_calls))
    if 'index_shape' in case:
        env['AFFON_CELL_INDEX_SHAPE'] = json.dumps(case['index_shape'])
    if 'index_axis' in case:
        env['AFFON_CELL_INDEX_AXIS'] = str(case['index_axis'])
    if 'classes' in case:
        env['AFFON_CELL_CLASSES'] = str(case['classes'])
    if 'target_shape' in case:
        env['AFFON_TARGET_SHAPE'] = json.dumps(case['target_shape'])
    if case['op'] == 'quantized_matmul':
        env['AFFON_Q_BITS'] = str(case['bits'])
        env['AFFON_Q_GROUP'] = str(case['group_size'])
    if case['op'] == 'topk':
        env['AFFON_TOPK_AXIS'] = str(case['axis'])
        env['AFFON_TOPK_K'] = str(case['k'])
        env['AFFON_TOPK_LARGEST'] = '1' if case.get('largest', True) else '0'
    if extra_env:
        env.update(extra_env)
    worker = 'affon-index-select.ts' if case['op'] == 'index_select' and len(case['shape']) == 3 else 'affon-topk.ts' if case['op'] == 'topk' else 'affon-quantized.ts' if case['op'] == 'quantized_matmul' else 'affon-scatter.ts' if case['op'] == 'scatter_add' else 'affon-dense-loss.ts' if case['op'] in {'cross_entropy', 'log_softmax_nll'} else 'affon-reduce-shape.ts' if case['op'] == 'reduce_to_shape' else 'affon-benchmark.ts'
    run = subprocess.run([str(binary.resolve()), str(ROOT / 'test/parity/workers' / worker)],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if run.returncode:
        raise RuntimeError(f"{case['id']}: Affon failed: {run.stderr[-4000:]}")
    lines = [line for line in run.stdout.splitlines() if line.startswith('{')]
    if len(lines) != 1:
        raise ValueError(f"{case['id']}: expected one Affon JSON result")
    result = json.loads(lines[0])
    if result.get("id") != case["id"] or result.get("shape") != case["shape"] or result.get("suite") != "affon_cell_samples" or result.get("kind") != kind:
        raise ValueError(f"{case['id']}: Affon result contract mismatch")
    return result


def pytorch(case, iterations, samples, kind="operator_peer"):
    import torch
    backend = case["id"].split("/")[0]
    device = {"metal": "mps", "cuda": "cuda"}.get(backend, "cpu")
    left, right = inputs(case)
    peer_dtype = torch.bfloat16 if case['dtype'] == 'bf16' else torch.float32
    x_base = torch.tensor(left, dtype=torch.long if case['op'] == 'one_hot' else peer_dtype, device=device)
    y_base = torch.tensor(right, dtype=torch.long if case['op'] in {'embedding', 'index_select', 'gather', 'cross_entropy_indexed', 'quantized_matmul', 'scatter_add'} else peer_dtype, device=device)
    if case['op'] == 'scatter_add':
        updates_base = torch.tensor(scatter_physical_parameters(case)[2], dtype=peer_dtype, device=device)
    if case['op'] == 'quantized_matmul':
        _, _, q_scales, q_biases = quantized_physical_parameters(case)
        scales_base = torch.tensor(q_scales, dtype=peer_dtype, device=device)
        biases_base = torch.tensor(q_biases, dtype=peer_dtype, device=device)
        shifts = torch.arange(32 // case['bits'], dtype=torch.long, device=device) * case['bits']
    mask_base = torch.tensor(mask_values(case), dtype=torch.long, device=device) if case['op'] in {'where', 'masked_fill'} else None
    op = case["op"]
    reduction_axis = 0 if ('/outer_axis' in case['id']) else 1
    output_width = case['shape'][-1] if case['shape'] else 1
    bias = (torch.tensor([col / 8 - output_width / 16 for col in range(output_width)],
                         dtype=peer_dtype, device=device)
            if kind == "program_peer" and op == "matmul" else None)
    def operation():
        if op == 'topk':
            layout = case['id'].split('/')[4].split('_')[0] if len(case['id'].split('/')) > 4 else 'strided' if case['id'].split('/')[3] == 'strided' else 'dense'
            x = x_base[1:, 1:] if layout == 'offset' else x_base[:, ::2] if layout == 'strided' else x_base
            values, indices = torch.topk(x, case['k'], dim=case['axis'], largest=case.get('largest', True))
            return (values + values if kind == 'program_peer' else values), indices
        if op == 'reduce_to_shape':
            x = x_base[1:, 1:] if '/offset_' in case['id'] else x_base
            if case['id'].split('/')[3] == 'strided' and '/offset_' not in case['id']:
                x = x_base[:, ::2]
            result = x.sum_to_size(*case['target_shape']) if case['target_shape'] else x.sum()
            return result + result if kind == 'program_peer' else result
        if op in {'cross_entropy', 'log_softmax_nll'}:
            offset = '/offset_' in case['id']
            cols = '/strided/cols_' in case['id']
            rows = '/strided/rows_' in case['id']
            x = x_base[1:, 1:] if offset else x_base[:, ::2] if cols else x_base[::2, :] if rows else x_base
            target = y_base[1:, 1:] if offset else y_base[:, ::2] if cols else y_base[::2, :] if rows else y_base
            result = torch.nn.functional.cross_entropy(x.float(), target.float()).to(peer_dtype).reshape(1)
            return result + result if kind == 'program_peer' else result
        if op == 'scatter_add':
            offset = '/offset_' in case['id']
            cols = '/strided/cols_' in case['id']
            rows = '/strided/rows_' in case['id']
            x = x_base[1:, 1:] if offset else x_base[:, ::2] if cols else x_base[::2, :] if rows else x_base
            i = y_base[1:, 1:] if offset else y_base[:, ::2] if cols else y_base[::2, :] if rows else y_base
            u = updates_base[1:, 1:] if offset else updates_base[:, ::2] if cols else updates_base[::2, :] if rows else updates_base
            result = x.scatter_add(case['index_axis'], i, u)
            return result + result if kind == 'program_peer' else result
        if op == 'quantized_matmul':
            transposed = '/transposed/' in case['id']
            offset = '/offset_' in case['id']
            xq = x_base.t() if transposed else x_base[1:, 1:] if offset else x_base
            wq = y_base[:, 1:] if transposed else y_base[1:, 1:] if offset else y_base
            sq = scales_base[1:, 1:] if offset else scales_base
            bq = biases_base[1:, 1:] if offset else biases_base
            digits = ((wq.unsqueeze(-1) >> shifts) & ((1 << case['bits']) - 1)).reshape(case['shape'][2], case['shape'][1])
            weight = digits.to(peer_dtype) * sq.repeat_interleave(case['group_size'], dim=1) + bq.repeat_interleave(case['group_size'], dim=1)
            result = xq @ weight.t()
            result = result.float()
            return result + result if kind == 'program_peer' else result
        x, y = (x_base[::2], y_base[::2]) if op == 'dot' and has_layout(case, 'strided') else (
            (x_base[1:], y_base[1:]) if op == 'dot' and has_layout(case, 'offset') else (
            (x_base.t(), y_base) if op == 'matmul' and '/transposed/lhs_' in case['id'] else (
            (x_base[:, ::2], y_base) if op == 'matmul' and '/strided/lhs_' in case['id'] else (
            (x_base, y_base.t()) if op == 'matmul' and '/transposed' in case['id'] else (
            (x_base, y_base[:, ::2]) if op == 'matmul' and has_layout(case, 'strided') else
            (x_base[:, ::2], y_base) if op in {'embedding', 'index_select', 'gather', 'cross_entropy_indexed'} and has_layout(case, 'strided') else
            (x_base[:, ::2], y_base[:, ::2]) if has_layout(case, 'strided') else (
            (x_base[1:, 1:], y_base if op in {'embedding', 'index_select', 'gather', 'cross_entropy_indexed'} else y_base[1:, 1:]) if has_layout(case, 'offset') else (x_base, y_base))
            )))))
        if op == 'mean_all' and '/rank1/' in case['id']:
            x = x.reshape(-1)
        result = x + y if op == "add" else x * y if op == "mul" else x - y if op == "sub" else x / y if op == "div" else (x == y).long() if op == "eq" else (x < y).long() if op == "lt" else (x > y).long() if op == "gt" else x.to(torch.bfloat16 if case['dtype'] == 'f32' else torch.float32) if op == 'cast' else -x if op == "neg" else torch.exp(x) if op == "exp" else torch.log(x) if op == "log" else torch.sqrt(x) if op == "sqrt" else torch.relu(x) if op == "relu" else torch.sigmoid(x) if op == "sigmoid" else torch.abs(x) if op == "abs" else torch.sign(x) if op == "sign" else torch.nn.functional.silu(x) if op == "silu" else torch.tanh(x) if op == "tanh" else torch.sin(x) if op == "sin" else torch.cos(x) if op == "cos" else torch.erf(x) if op == 'erf' else torch.nn.functional.gelu(x, approximate='tanh') if op == 'gelu' else torch.clamp(x, 1.25, 2.25) if op == 'clamp' else torch.nn.functional.layer_norm(x, case["shape"] if is_outer_axis(case) else [case["shape"][1]]) if op == "layer_norm" else torch.nn.functional.log_softmax(x.float(), dim=reduction_axis).to(peer_dtype) if op == "log_softmax" else torch.nn.functional.rms_norm(x.float(), case["shape"] if is_outer_axis(case) else [case["shape"][1]], eps=1e-5).to(peer_dtype) if op == "rms_norm" else (x.t().reshape(case["shape"][0], case["shape"][1]) if case["id"].endswith("/materialized") else x.reshape(case["shape"][1], case["shape"][0])) if op == "reshape" else x.t() if op == "transpose" else x.permute(1, 0) if op == "permute" else x[:case["shape"][0] // 2, :case["shape"][1] // 2] if op == "slice" else (x.contiguous() if case["id"].endswith("/view_only") else x.t().contiguous()) if op == "contiguous" else torch.cat([x, y], dim=1) if op == "cat" else torch.stack([x, y], dim=0) if op == "stack" else x.squeeze(0) if op == "squeeze" else x.unsqueeze(0) if op == "unsqueeze" else torch.nn.functional.embedding(y, x) if op == 'embedding' else torch.index_select(x, case.get('index_axis', 0), y) if op == 'index_select' else torch.gather(x, case.get('index_axis', 1), y) if op == 'gather' else torch.where(mask_base.bool(), x, y) if op == 'where' else x.masked_fill(mask_base.bool(), -2) if op == 'masked_fill' else ((x.unsqueeze(-1) == torch.arange(case['classes'], device=device)).float() if '/invalid_index' in case['id'] else torch.nn.functional.one_hot(x.long(), case['classes']).float()) if op == 'one_hot' else torch.nn.functional.cross_entropy(x.float(), y).to(peer_dtype) if op == 'cross_entropy_indexed' else x @ y if op == "matmul" else torch.dot(x, y) if op == "dot" else torch.argmin(x) if op == "argmin_all" else torch.argmax(x) if op == "argmax_all" else torch.argmin(x, dim=reduction_axis) if op == "argmin_axis" else torch.argmax(x, dim=reduction_axis) if op == "argmax_axis" else x.sum() if op == "sum_all" else x.mean() if op == "mean_all" else x.min() if op == "min_all" else x.max() if op == "max_all" else x.var(correction=0) if op == "variance_all" else x.std(correction=0) if op == "std_all" else x.sum(dim=reduction_axis) if op == "sum_axis" else x.mean(dim=reduction_axis) if op == "mean_axis" else x.min(dim=reduction_axis).values if op == "min_axis" else x.max(dim=reduction_axis).values if op == "max_axis" else (x.float().var(dim=reduction_axis, correction=0).to(peer_dtype) if case['dtype'] == 'bf16' else x.var(dim=reduction_axis, correction=0)) if op == "variance_axis" else (x.float().std(dim=reduction_axis, correction=0).to(peer_dtype) if case['dtype'] == 'bf16' else x.std(dim=reduction_axis, correction=0)) if op == "std_axis" else torch.softmax(x, dim=reduction_axis)
        return result + bias if bias is not None else (result + result if kind == "program_peer" else result).reshape(1) if op in {"sum_all", "mean_all", "min_all", "max_all", "variance_all", "std_all", "argmin_all", "argmax_all", "cross_entropy_indexed"} else result + result if kind == "program_peer" else result
    def run():
        result = operation()
        if op == 'topk':
            values, indices = result
            return flatten(values.float().tolist()) + flatten(indices.tolist()), list(values.shape), f'pair:{case["dtype"]},i64'
        return flatten(result.float().tolist()), list(result.shape), {'torch.float32': 'f32', 'torch.bfloat16': 'bf16', 'torch.int64': 'i64'}[str(result.dtype)]
    output, output_shape, output_dtype = run()
    indices = sorted(set([0, len(output) // 4, len(output) // 2, len(output) * 3 // 4, len(output) - 1]))
    output_samples = [{"index": index, "value": output[index]} for index in indices]
    output_checksum = sum(output)
    for _ in range(20):
        run()
    samples_ms = []
    execution_ms, readback_ms = [], []
    for _ in range(samples):
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        execution = readback = 0
        for _ in range(iterations):
            phase = time.perf_counter_ns()
            result = operation()
            execution += time.perf_counter_ns() - phase
            phase = time.perf_counter_ns()
            if op == 'topk':
                result[0].tolist(); result[1].tolist()
            else:
                result.tolist()
            readback += time.perf_counter_ns() - phase
        samples_ms.append((time.perf_counter_ns() - start) / iterations / 1e6)
        execution_ms.append(execution / iterations / 1e6)
        readback_ms.append(readback / iterations / 1e6)
    return {"output_count": len(output), "output_shape": output_shape, "output_dtype": output_dtype,
            **({'output_hash': topk_output_hash(output)} if op == 'topk' else {}),
            **({'outputs': [{'shape': output_shape, 'dtype': case['dtype'], 'count': len(output) // 2},
                             {'shape': output_shape, 'dtype': 'i64', 'count': len(output) // 2}]} if op == 'topk' else {}),
            "output_samples": output_samples,
            "output_checksum": output_checksum}, {"total_ms": samples_ms, "execution_ms": execution_ms, "readback_ms": readback_ms}


def one(binary, case, iterations, samples, rounds, target, kind="operator_peer"):
    import torch
    left, right = inputs(case)
    mask = mask_values(case) if case['op'] in {'where', 'masked_fill'} else None
    q_extra = list(quantized_physical_parameters(case)[2:]) if case['op'] == 'quantized_matmul' else [scatter_physical_parameters(case)[2]] if case['op'] == 'scatter_add' else []
    fixture_values = [left] if case['op'] == 'reduce_to_shape' else [left, right, *q_extra, mask] if mask is not None else [left, right, *q_extra]
    fixture = hashlib.sha256(json.dumps(fixture_values, separators=(",", ":")).encode()).hexdigest()
    bias_fixture = ([col / 8 - case["shape"][-1] / 16 for col in range(case["shape"][-1])]
                    if kind == "program_peer" and case["op"] == "matmul" else None)
    measurements = []
    for round_index in range(rounds):
        if round_index % 2:
            peer_output, peer_timing = pytorch(case, iterations, samples, kind)
            a = affon(binary, case, iterations, samples, kind)
            order = ["pytorch", "affon"]
        else:
            a = affon(binary, case, iterations, samples, kind)
            peer_output, peer_timing = pytorch(case, iterations, samples, kind)
            order = ["affon", "pytorch"]
        if a["output_count"] != peer_output["output_count"] or a.get("output_shape") != peer_output["output_shape"] or a.get("output_dtype") != peer_output["output_dtype"] or a.get('outputs') != peer_output.get('outputs') or [item["index"] for item in a["output_samples"]] != [item["index"] for item in peer_output["output_samples"]]:
            raise ValueError(f"{case['id']}: output length mismatch")
        error = max(abs(x["value"] - y["value"]) for x, y in zip(a["output_samples"], peer_output["output_samples"]))
        hash_matches = case['op'] != 'topk' or a.get('output_hash') == peer_output.get('output_hash')
        relative_tolerance = 0.02 if case['dtype'] == 'bf16' else 1e-5 if case['op'] in {'variance_axis', 'std_axis', 'quantized_matmul'} else 2e-6 if case['op'] in {'dot', 'layer_norm', 'exp'} else 1e-6
        tolerance = relative_tolerance * max(1, max(abs(x["value"]) for x in peer_output["output_samples"]))
        checksum_error = abs(a["output_checksum"] - peer_output["output_checksum"])
        checksum_tolerance = tolerance * peer_output["output_count"]
        measurements.append({"round": round_index, "order": order, "affon_samples_ms": a["samples_ms"],
                             "peer_samples_ms": peer_timing["total_ms"],
                             "affon_execution_ms": a["execution_ms"], "affon_readback_ms": a["readback_ms"],
                             "peer_execution_ms": peer_timing["execution_ms"], "peer_readback_ms": peer_timing["readback_ms"],
                             "max_abs_error": error, "output_hash_matches": hash_matches,
                             "checksum_error": checksum_error,
                             "selected_path": a["selected_path"]})
    if len({json.dumps(row["selected_path"], sort_keys=True) for row in measurements}) != 1:
        raise ValueError(f"{case['id']}: Affon selected path changed across rounds")
    affon_samples = [sample for row in measurements for sample in row["affon_samples_ms"]]
    peer_samples = [sample for row in measurements for sample in row["peer_samples_ms"]]
    error = max(row["max_abs_error"] for row in measurements)
    checksum_error = max(row["checksum_error"] for row in measurements)
    hashes_match = all(row['output_hash_matches'] for row in measurements)
    return {"schema_version": 1, "suite": "affon_cell_benchmark", "cell_id": case["id"], "kind": kind,
            "target": target, "affon_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "peer_engine": "pytorch", "peer_version": torch.__version__, "selected_path": measurements[0]["selected_path"],
                         "contract": {"shape": case["shape"], "dtype": case["dtype"], "layout": "transposed_input_offset_weights" if case['op'] == 'quantized_matmul' and '/transposed/' in case['id'] else "offset_all_four_inputs" if case['op'] == 'quantized_matmul' and '/offset_' in case['id'] else "stride_2_vector" if case['op'] == 'dot' and has_layout(case, 'strided') else "offset_1_vector" if case['op'] == 'dot' and has_layout(case, 'offset') else "transposed_lhs" if '/transposed/lhs_' in case['id'] else "stride_2_lhs_columns" if '/strided/lhs_' in case['id'] else "transposed_rhs" if '/transposed' in case['id'] else "stride_2_rhs_columns" if case['op'] == 'matmul' and has_layout(case, 'strided') else "stride_2_columns" if has_layout(case, 'strided') else "offset_1_1" if has_layout(case, 'offset') else "dense_zero_offset_broadcast_rhs" if '/broadcast' in case['id'] else "dense_zero_offset",
                         "input_shapes": [shape_of(left)] if case['op'] in {'reduce_to_shape', 'topk'} else [shape_of(left), shape_of(right), *[shape_of(value) for value in q_extra], *([shape_of(mask)] if mask is not None else [])],
                         **({"target_shape": case['target_shape']} if case['op'] == 'reduce_to_shape' else {}),
                         "output_shape": peer_output["output_shape"], "output_dtype": peer_output["output_dtype"],
                         **({'outputs': peer_output['outputs']} if case['op'] == 'topk' else {}),
                         "compile_options": {"dense_inputs": bool(case.get("dense_inputs", False))},
                         **({"program": program_graph(case),
                             **({"extra_input_sha256": hashlib.sha256(json.dumps(bias_fixture, separators=(",", ":")).encode()).hexdigest()}
                                if bias_fixture is not None else {})}
                            if kind == "program_peer" else {}),
                         "timing_scope": ("resident quantized matmul with PyTorch GPU bit unpack, affine dequantization and matmul; synchronized host materialization included; diagnostic cross-implementation timing"
                                          if case['op'] == 'quantized_matmul' else
                                          "resident two-operation Program, synchronized full host materialization, output allocation and disposal included"
                                          if kind == "program_peer" else
                                          "resident operation, synchronized full host materialization, output allocation and disposal included"),
                         "optimization_level": "safe", "input_sha256": fixture,
                         "iterations_per_sample": iterations, "warmup_calls": 20,
                         "affon_clock": "performance.now", "peer_clock": "time.perf_counter_ns",
                         "rounds": rounds},
            "correctness": {"passed": error <= tolerance and checksum_error <= checksum_tolerance and hashes_match,
                            **({'output_hash_matches': hashes_match} if case['op'] == 'topk' else {}),
                            "max_abs_error": error, "tolerance": tolerance,
                            "checksum_error": checksum_error, "checksum_tolerance": checksum_tolerance,
                            "checked_elements": len(peer_output["output_samples"]),
                            "total_elements": peer_output["output_count"]},
            "rounds": measurements, "affon_samples_ms": affon_samples, "peer_samples_ms": peer_samples,
            "phase_medians_ms": {key: statistics.median(sample for row in measurements for sample in row[key])
                                 for key in ("affon_execution_ms", "affon_readback_ms", "peer_execution_ms", "peer_readback_ms")},
            "ratio_of_medians": statistics.median(affon_samples) / statistics.median(peer_samples),
            "paired_round_ratios": [statistics.median(row["affon_samples_ms"]) / statistics.median(row["peer_samples_ms"])
                                    for row in measurements]}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--catalog", type=Path, default=ROOT / "test/benchmark/capability-probes.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated case IDs")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--kind", choices=["operator_peer", "program_peer"], default="operator_peer")
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1 or args.rounds < 2:
        parser.error("need at least two rounds, three samples per round, and one operation per sample")
    if not args.binary.is_file():
        parser.error("Affon binary missing")
    catalog = json.loads(args.catalog.read_text())
    cases = [case for case in catalog["cases"] if not args.only or case["id"] in args.only.split(",")]
    if args.only and len(cases) != len(set(args.only.split(","))):
        parser.error("unknown or duplicate case ID")
    if any(case["id"].startswith("metal/") for case in cases) and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS unavailable")
    if any(case["id"].startswith("cuda/") for case in cases) and not torch.cuda.is_available():
        parser.error("PyTorch CUDA unavailable")
    if any(case["id"].startswith("cuda/") for case in cases):
        model = torch.cuda.get_device_name()
        target = f"{model}/{platform.release()}/{platform.machine()}"
    else:
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        target = f"{model}/{platform.mac_ver()[0]}/{platform.machine()}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        report = one(args.binary, case, args.iterations, args.samples, args.rounds, target, args.kind)
        path = args.output_dir / (case["id"].replace("/", "__") + ".json")
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case['id']}: ratio {report['ratio_of_medians']:.2f}, max error {report['correctness']['max_abs_error']:.3g}", flush=True)


if __name__ == "__main__":
    main()
