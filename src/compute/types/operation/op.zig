const std = @import("std");
const Tensor = @import("../tensor/tensor.zig").Tensor;
const OpTag = @import("tag.zig").OpTag;
const OpOptions = @import("options.zig").OpOptions;
const ExecutionMetadata = @import("execution_metadata.zig").ExecutionMetadata;

pub const Op = struct {
    tag: OpTag,
    inputs: []const *Tensor,
    options: OpOptions = .{ .none = {} },
    execution_metadata: ExecutionMetadata = .{},

    pub fn init(tag: OpTag, inputs: []const *Tensor, options: OpOptions) !Op {
        return initWithExecutionMetadata(tag, inputs, options, .{});
    }

    pub fn initWithExecutionMetadata(
        tag: OpTag,
        inputs: []const *Tensor,
        options: OpOptions,
        execution_metadata: ExecutionMetadata,
    ) !Op {
        const op = Op{
            .tag = tag,
            .inputs = inputs,
            .options = options,
            .execution_metadata = execution_metadata,
        };
        try op.validate();
        return op;
    }

    pub fn validate(self: Op) !void {
        switch (self.tag) {
            .add, .sub, .mul, .div, .eq, .lt, .gt => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .binary => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .abs, .exp, .log, .neg, .sqrt, .sign, .relu, .sigmoid, .silu, .tanh, .gelu, .gelu_grad => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .unary => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .where => {
                if (self.inputs.len != 3) return error.InvalidInputCount;
                switch (self.options) {
                    .none => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .masked_fill => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .masked_fill => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .softmax => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .softmax => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .log_softmax => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .log_softmax => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .log_softmax_nll => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .log_softmax_nll => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .cross_entropy_indexed => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .cross_entropy_indexed => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .cross_entropy_indexed_backward => {
                if (self.inputs.len != 3) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .cross_entropy_indexed_backward => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .cross_entropy => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .cross_entropy => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .cast => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .cast => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .layer_norm => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .layer_norm => |ln| {
                        if (!(ln.eps > 0.0)) return error.InvalidEpsilon;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .rms_norm => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .rms_norm => |rn| {
                        if (!(rn.eps > 0.0)) return error.InvalidEpsilon;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .clamp => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .clamp => |clamp| {
                        if (clamp.min > clamp.max) return error.InvalidClampBounds;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .sum_all, .mean_all, .min_all, .max_all, .variance_all, .std_all, .argmin_all, .argmax_all => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .reduce_all => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .sum_axis, .mean_axis, .min_axis, .max_axis, .variance_axis, .std_axis, .argmin_axis, .argmax_axis => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .reduce_axis => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .cat => {
                if (self.inputs.len < 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .concat => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .stack => {
                if (self.inputs.len < 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .stack => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .contiguous => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .reshape => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .reshape => |reshape| {
                        if (reshape.shape.len == 0) return error.InvalidShape;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .slice => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .slice => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .gather => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .gather => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .embedding => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .embedding => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .index_select => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .index_select => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .scatter_add => {
                if (self.inputs.len != 3) return error.InvalidInputCount;
                switch (self.options) {
                    .scatter_add => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .topk => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .topk => |topk| {
                        if (topk.k == 0) return error.InvalidTopK;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .one_hot => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .one_hot => |one_hot| {
                        if (one_hot.num_classes == 0) return error.InvalidClassCount;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .reduce_to_shape => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .reduce_to_shape => |shape_opt| {
                        if (shape_opt.shape.len == 0) return error.InvalidShape;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .transpose => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .transpose => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .permute => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .permute => |permute| {
                        if (permute.axes.len == 0) return error.InvalidPermutation;
                    },
                    else => return error.InvalidOpOptions,
                }
            },
            .squeeze => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .none, .squeeze => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .unsqueeze => {
                if (self.inputs.len != 1) return error.InvalidInputCount;
                switch (self.options) {
                    .unsqueeze => {},
                    else => return error.InvalidOpOptions,
                }
            },
            .dot, .matmul => {
                if (self.inputs.len != 2) return error.InvalidInputCount;
                switch (self.options) {
                    .none => {},
                    else => return error.InvalidOpOptions,
                }
            },
        }
    }
};

test "op init validates options" {
    const allocator = std.testing.allocator;
    const value = try Tensor.fromSliceF32(allocator, &.{2}, &.{ 1, 2 });
    defer value.deinit();

    _ = try Op.init(.neg, &.{value}, .{ .unary = .{} });
}
