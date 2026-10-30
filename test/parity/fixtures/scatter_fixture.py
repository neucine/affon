"""Deterministic scatter-add fixture and independent numerical reference."""


def logical_parameters(case):
    if len(case['shape']) == 3:
        batches, rows, cols = case['shape']
        axis = case['index_axis']
        index_shape = case['index_shape']
        base = [[[(batch * 5 + row * 3 + col) / 8 for col in range(cols)]
                 for row in range(rows)] for batch in range(batches)]
        indices = [[[(batch * 7 + row * 3 + col * 5) % case['shape'][axis]
                     for col in range(index_shape[2])] for row in range(index_shape[1])]
                   for batch in range(index_shape[0])]
        updates = [[[(batch + row + col * 2) % 7 / 8 - 3 / 8
                     for col in range(index_shape[2])] for row in range(index_shape[1])]
                   for batch in range(index_shape[0])]
        return base, indices, updates
    rows, cols = case['shape']
    axis = case['index_axis']
    index_shape = case['index_shape']
    base = [[(row * 3 + col) / 8 for col in range(cols)] for row in range(rows)]
    indices = [[(row * 3 + col * 5) % (rows if axis == 0 else cols)
                for col in range(index_shape[1])] for row in range(index_shape[0])]
    if '/invalid_index' in case['id']:
        indices[0][0] = case['shape'][axis] + 1
    updates = [[((row + col * 2) % 7 - 3) / 8
                for col in range(index_shape[1])] for row in range(index_shape[0])]
    return base, indices, updates


def physical_parameters(case):
    values = logical_parameters(case)
    if len(case['shape']) == 3:
        return values
    if '/strided/cols_' in case['id']:
        return tuple([[item for value in row for item in (value, 0)] for row in matrix]
                     for matrix in values)
    if '/strided/rows_' in case['id']:
        return tuple([row for original in matrix for row in (original, [0] * len(original))]
                     for matrix in values)
    if '/offset_' not in case['id']:
        return values
    def pad(matrix):
        return [[0] * (len(matrix[0]) + 1), *[[0, *row] for row in matrix]]
    return tuple(pad(matrix) for matrix in values)


def reference(case):
    base, indices, updates = logical_parameters(case)
    if len(case['shape']) == 3:
        output = [[row[:] for row in batch] for batch in base]
        for batch in range(len(indices)):
            for row in range(len(indices[batch])):
                for col in range(len(indices[batch][row])):
                    target = [batch, row, col]
                    target[case['index_axis']] = indices[batch][row][col]
                    output[target[0]][target[1]][target[2]] += updates[batch][row][col]
        return [value for batch in output for row in batch for value in row]
    output = [row[:] for row in base]
    for row in range(len(indices)):
        for col in range(len(indices[row])):
            target = (indices[row][col], col) if case['index_axis'] == 0 else (row, indices[row][col])
            if target[case['index_axis']] >= case['shape'][case['index_axis']]:
                continue
            output[target[0]][target[1]] += updates[row][col]
    return [value for row in output for value in row]
