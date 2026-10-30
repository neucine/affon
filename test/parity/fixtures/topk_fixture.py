"""Stable, tie-free input and independent top-k reference."""
import struct


def output_hash(values):
    result = 2166136261
    for value in values:
        result = ((result ^ (round(value * 8) & 0xffffffff)) * 16777619) & 0xffffffff
    return result


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def physical_values(case):
    if len(case['shape']) == 3:
        batches, rows, cols = case['shape']
        rounded = bf16 if case['dtype'] == 'bf16' else lambda value: value
        return [[[rounded(((batch * 31 + row * 97 + col * 53) % 1009) / 8)
                  for col in range(cols)] for row in range(rows)]
                for batch in range(batches)]
    rows, cols = case['shape']
    layout = case['id'].split('/')[4].split('_')[0] if len(case['id'].split('/')) > 4 else 'strided' if case['id'].split('/')[3] == 'strided' else 'dense'
    physical_rows = rows + (layout == 'offset')
    physical_cols = cols * 2 if layout == 'strided' else cols + (layout == 'offset')
    rounded = bf16 if case['dtype'] == 'bf16' else lambda value: value
    return [[rounded(((row * 97 + col * 53) % 1009) / 8) for col in range(physical_cols)]
            for row in range(physical_rows)]


def logical_values(case):
    values = physical_values(case)
    if len(case['shape']) == 3:
        return values
    layout = case['id'].split('/')[4].split('_')[0] if len(case['id'].split('/')) > 4 else 'strided' if case['id'].split('/')[3] == 'strided' else 'dense'
    if layout == 'offset':
        return [row[1:] for row in values[1:]]
    if layout == 'strided':
        return [row[::2] for row in values]
    return values


def reference(case):
    values = logical_values(case)
    if len(case['shape']) == 3:
        shape = case['shape']
        axis, k = case['axis'], case['k']
        output_shape = shape.copy()
        output_shape[axis] = k
        output_values = []
        output_indices = []
        for batch in range(output_shape[0]):
            for row in range(output_shape[1]):
                for col in range(output_shape[2]):
                    position = [batch, row, col]
                    candidates = []
                    for index in range(shape[axis]):
                        location = position.copy()
                        location[axis] = index
                        candidates.append((values[location[0]][location[1]][location[2]], index))
                    largest = case.get('largest', True)
                    selected = sorted(candidates, key=lambda pair: (-pair[0] if largest else pair[0], pair[1]))
                    value, index = selected[position[axis]]
                    output_values.append(value)
                    output_indices.append(index)
        return output_values + output_indices
    rows, cols = case['shape']
    axis, k = case['axis'], case['k']
    output_values = [[0] * (k if axis == 1 else cols) for _ in range(rows if axis == 1 else k)]
    output_indices = [[0] * len(output_values[0]) for _ in output_values]
    for outer in range(rows if axis == 1 else cols):
        candidates = [(values[outer][inner] if axis == 1 else values[inner][outer], inner)
                      for inner in range(cols if axis == 1 else rows)]
        largest = case.get('largest', True)
        for rank, (value, index) in enumerate(sorted(candidates, key=lambda pair: (-pair[0] if largest else pair[0], pair[1]))[:k]):
            position = (outer, rank) if axis == 1 else (rank, outer)
            output_values[position[0]][position[1]] = value
            output_indices[position[0]][position[1]] = index
    return [item for row in output_values for item in row] + [item for row in output_indices for item in row]
