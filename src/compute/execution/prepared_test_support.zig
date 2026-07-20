const std = @import("std");
const Tensor = @import("../shared/types/tensor/tensor.zig").Tensor;
const OpTag = @import("../shared/types/operation/tag.zig").OpTag;
const OpOptions = @import("../shared/types/operation/options.zig").OpOptions;
const BroadcastSpec = @import("../shared/types/ir/plan.zig").BroadcastSpec;
const semantic = @import("../shared/sema/index.zig");

pub fn broadcast(
    allocator: std.mem.Allocator,
    tag: OpTag,
    values: []const *const Tensor,
    options: OpOptions,
) !BroadcastSpec {
    var specs: [3]@import("../shared/types/tensor/tensor_spec.zig").TensorSpec = undefined;
    if (values.len > specs.len) return error.InvalidExecutionPlan;
    for (values, 0..) |value, i| specs[i] = try value.spec();
    var inferred = try semantic.inferFromSpecs(allocator, tag, specs[0..values.len], options);
    defer inferred.deinit();
    return inferred.broadcast orelse error.InvalidExecutionPlan;
}
