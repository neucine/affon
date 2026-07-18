const std = @import("std");
const Tensor = @import("../../types/tensor/tensor.zig").Tensor;
const Op = @import("../../types/operation/op.zig").Op;
const semantic = @import("../../plan/sema/index.zig");
const planning = @import("../../plan/eager.zig");
const runner = @import("runner.zig");

pub fn createPlan(allocator: std.mem.Allocator, op: Op) !planning.Plan {
    var info = try semantic.infer(allocator, op);
    defer info.deinit();
    return planning.create(allocator, op, info);
}

pub fn executeAll(allocator: std.mem.Allocator, op: Op) !runner.ExecutionResult {
    var plan = try createPlan(allocator, op);
    defer plan.deinit();
    return runner.executeAllWithPlan(allocator, op, &plan);
}

pub fn execute(allocator: std.mem.Allocator, op: Op) !*Tensor {
    var result = try executeAll(allocator, op);
    if (result.secondary != null) {
        result.deinit();
        return error.MultiOutputRequiresExecuteAll;
    }
    const output = result.primary;
    result.primary = undefined;
    return output;
}
