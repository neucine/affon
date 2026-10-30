"""Deterministic soft-target cross entropy fixture and scalar reference."""
import math
import struct


def parameters(case):
    if len(case['shape']) == 3:
        batches, classes, spatial = case['shape']
        logits = [[[(batch * 5 + position * 3 + col) / 8 for position in range(spatial)]
                   for col in range(classes)] for batch in range(batches)]
        targets = [[[0.5 if col in ((batch + position * 3 + 1) % classes,
                                    (batch + position * 3 + 2) % classes) else 0.0
                     for position in range(spatial)] for col in range(classes)]
                   for batch in range(batches)]
        return logits, targets
    rows, classes = case['shape']
    logits = [[(row * 3 + col) / 8 for col in range(classes)] for row in range(rows)]
    targets = [[0.5 if col in ((row * 3 + 1) % classes, (row * 3 + 2) % classes) else 0.0
                for col in range(classes)] for row in range(rows)]
    return logits, targets


def physical_parameters(case):
    logits, targets = parameters(case)
    if len(case['shape']) == 3:
        return logits, targets
    if '/strided/cols_' in case['id']:
        return tuple([[item for value in row for item in (value, 0.0)] for row in matrix]
                     for matrix in (logits, targets))
    if '/strided/rows_' in case['id']:
        return tuple([row for original in matrix for row in (original, [0.0] * len(original))]
                     for matrix in (logits, targets))
    if '/offset_' not in case['id']:
        return logits, targets
    def pad(values):
        return [[0.0] * (len(values[0]) + 1), *[[0.0, *row] for row in values]]
    return pad(logits), pad(targets)


def reference(case):
    logits, targets = parameters(case)
    loss = 0.0
    if len(case['shape']) == 3:
        batches, classes, spatial = case['shape']
        pairs = [([logits[b][c][s] for c in range(classes)],
                  [targets[b][c][s] for c in range(classes)])
                 for b in range(batches) for s in range(spatial)]
    else:
        pairs = list(zip(logits, targets))
    for row, weights in pairs:
        maximum = max(row)
        normalizer = math.log(sum(math.exp(value - maximum) for value in row))
        loss -= sum(weight * (value - maximum - normalizer) for weight, value in zip(weights, row))
    result = loss / len(pairs)
    if case['dtype'] == 'bf16':
        bits = struct.unpack('<I', struct.pack('<f', result))[0]
        bits += 0x7fff + ((bits >> 16) & 1)
        result = struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]
    return [result]
