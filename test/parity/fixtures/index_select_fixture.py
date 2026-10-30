"""Independent rank-three index selection fixture and reference."""
import struct


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def parameters(case):
    batches, rows, cols = case['shape']
    rounded = bf16 if case['dtype'] == 'bf16' else lambda value: value
    values = [[[rounded(((batch * 31 + row * 97 + col * 53) % 1009) / 8)
                for col in range(cols)] for row in range(rows)]
              for batch in range(batches)]
    axis = case['index_axis']
    indices = [(index * 3 + 1) % case['shape'][axis]
               for index in range(case['index_shape'][0])]
    return values, indices


def reference(case):
    values, indices = parameters(case)
    shape = case['shape'].copy()
    shape[case['index_axis']] = len(indices)
    result = []
    for batch in range(shape[0]):
        for row in range(shape[1]):
            for col in range(shape[2]):
                position = [batch, row, col]
                position[case['index_axis']] = indices[position[case['index_axis']]]
                result.append(values[position[0]][position[1]][position[2]])
    return result
