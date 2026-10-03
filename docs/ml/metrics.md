# Metrics

Metrics evaluate model outputs; they do not update model state. The canonical
Program API currently has no separate metrics module. Express a reusable metric
as a Program from `affon:ops`, or calculate a reporting value from evaluated
tensors after a model run.

This distinction matters:

- a **loss Program** is the scalar objective passed to `gradient(...)` or
  `optimize(...)`;
- a **metric** is reporting computation and need not be differentiable;
- an immediate `affon:ops` call computes a value in the operands' Session and
  never creates an eager gradient tape.

Keep metrics outside an optimized loss Program unless they are genuinely part
of the training objective. Dispose metric result tensors after reading them.

Legacy metric helpers that operate on the old eager tensor surface belong to
the legacy compatibility layer and should not be used to explain new Programs.
