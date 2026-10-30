"""Shared deterministic four-input quantized matmul fixture and peer operation."""


def parameters(case):
    rows, inner, outputs = case['shape']
    bits, group = case['bits'], case['group_size']
    per_word = 32 // bits
    mask = (1 << bits) - 1
    x = [[((row * 3 + col) % 9 - 4) / 8 for col in range(inner)] for row in range(rows)]
    q = [[(out * 3 + col * 5) & mask for col in range(inner)] for out in range(outputs)]
    packed = [[sum(q[out][word * per_word + lane] << (lane * bits)
                   for lane in range(per_word)) for word in range(inner // per_word)]
              for out in range(outputs)]
    scales = [[(1 + (out + block) % 3) / 4 for block in range(inner // group)]
              for out in range(outputs)]
    biases = [[((out + block) % 3 - 1) / 8 for block in range(inner // group)]
              for out in range(outputs)]
    return x, packed, scales, biases


def physical_parameters(case):
    x, packed, scales, biases = parameters(case)
    transposed = '/transposed/' in case['id']
    offset = '/offset_' in case['id']
    if transposed and offset:
        raise ValueError('ambiguous quantized layout')
    def pad(values):
        return [[0] * (len(values[0]) + 1), *[[0, *row] for row in values]]
    if transposed:
        return [list(col) for col in zip(*x)], [[0, *row] for row in packed], scales, biases
    if offset:
        return pad(x), pad(packed), pad(scales), pad(biases)
    return x, packed, scales, biases


def reference(case):
    x, packed, scales, biases = parameters(case)
    rows, inner, outputs = case['shape']
    bits, group = case['bits'], case['group_size']
    per_word = 32 // bits
    mask = (1 << bits) - 1
    return [sum(x[row][col] * (
        ((packed[out][col // per_word] >> ((col % per_word) * bits)) & mask)
        * scales[out][col // group] + biases[out][col // group])
        for col in range(inner)) for row in range(rows) for out in range(outputs)]
