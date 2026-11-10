"""PyTorch comparison for Affon's value-order then lower-index top-k contract."""


def stable_topk(x, k, axis, largest=True):
    # Native torch.topk does not promise stable tied indices. A stable sort
    # implements the required semantics for all inputs, including cutoff ties.
    values, indices = x.sort(dim=axis, descending=largest, stable=True)
    return values.narrow(axis, 0, k), indices.narrow(axis, 0, k)
