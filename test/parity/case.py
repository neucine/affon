"""One vocabulary for the dimensions of a parity case.

Old catalogs keep their IDs and fields. This module exposes their dimensions
explicitly in new run records without changing saved results.
"""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def operation_family(operation, regime_group=None):
    if operation in {'add', 'sub', 'mul', 'div'}:
        return 'arithmetic'
    if operation in {'eq', 'lt', 'gt', 'where', 'masked_fill'}:
        return 'comparison'
    if operation in {'dot', 'matmul'}:
        return 'matrix'
    if operation == 'quantized_matmul':
        return 'quantized_matrix'
    if operation in {'softmax', 'log_softmax', 'layer_norm', 'rms_norm'}:
        return 'normalization'
    if operation.endswith(('_all', '_axis')) or operation == 'reduce_to_shape':
        return 'reduction'
    if operation in {'gather', 'embedding', 'index_select', 'scatter_add', 'topk', 'one_hot'}:
        return 'indexing'
    if operation in {'cross_entropy', 'cross_entropy_indexed', 'cross_entropy_indexed_backward', 'log_softmax_nll'}:
        return 'loss'
    if operation in {'cat', 'stack', 'contiguous', 'reshape', 'slice', 'permute', 'transpose', 'squeeze', 'unsqueeze'}:
        return 'structural'
    if operation in {'abs', 'exp', 'log', 'neg', 'sqrt', 'sign', 'relu', 'sigmoid', 'silu',
                     'tanh', 'erf', 'gelu', 'sin', 'cos', 'gelu_grad', 'clamp', 'cast'}:
        return 'elementwise_math'
    if regime_group == 'elementwise':
        return 'elementwise_math'
    return regime_group


def families():
    policy = json.loads((ROOT / 'test/benchmarks/coverage-policy.json').read_text())
    return {operation: operation_family(operation, group)
            for operation, group in policy['op_regime'].items()}


def layout_of(case):
    parts = case['id'].split('/')[3:]
    words = '_'.join(parts).split('_')
    inferred = 'dense'
    for layout in ('offset', 'strided', 'transposed'):
        if layout in words or layout in parts:
            inferred = layout
            break
    declared = case.get('layout', inferred)
    if not isinstance(declared, str) or not declared or (inferred != 'dense' and declared != inferred):
        raise ValueError(f"{case['id']}: layout and ID disagree")
    return declared


def normalize(case, operation_families=None):
    parts = case['id'].split('/')
    if len(parts) < 4 or parts[0] not in ('cpu', 'metal', 'cuda'):
        raise ValueError(f"invalid case ID: {case['id']}")
    device, dtype, operation, regime, *variant = parts
    if case.get('dtype') != dtype or case.get('op') != operation:
        raise ValueError(f"{case['id']}: ID and case fields disagree")
    expected_family = (operation_families or {}).get(operation) or operation_family(operation)
    family = case.get('family') or expected_family
    if not family:
        raise ValueError(f"{case['id']}: operation family missing")
    if expected_family and family != expected_family:
        raise ValueError(f"{case['id']}: operation family and operation disagree")
    scope = case.get('scope', 'operation')
    if scope not in ('operation', 'program', 'model_piece'):
        raise ValueError(f"{case['id']}: invalid execution scope")
    piece = case.get('piece')
    if scope != 'operation' and not piece:
        raise ValueError(f"{case['id']}: Program or model piece identity missing")
    if scope == 'operation' and piece:
        raise ValueError(f"{case['id']}: operation case cannot name a Program or model piece")
    return {'device': device, 'dtype': dtype, 'operation_family': family,
            'operation': operation, 'shape': case['shape'],
            'layout': layout_of(case), 'regime': regime,
            'variant': '/'.join(variant) or None, 'scope': scope,
            'piece': piece}


def contract_hash(case):
    """Hash operation arguments; normalized identity fields are compared separately."""
    arguments = {key: value for key, value in case.items()
                 if key not in {'family', 'layout', 'scope', 'piece'}}
    encoded = json.dumps(arguments, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()
