"""PyTorch scatter-add with Affon's skip-out-of-range update semantics."""


def scatter_add_skip_invalid(x, indices, updates, axis):
    valid = (indices >= 0) & (indices < x.shape[axis])
    safe_indices = indices.masked_fill(~valid, 0)
    safe_updates = updates.masked_fill(~valid, 0)
    return x.scatter_add(axis, safe_indices, safe_updates)
