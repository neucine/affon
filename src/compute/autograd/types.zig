const Value = @import("../types/tensor/value.zig").Value;
const OpTag = @import("../types/operation/tag.zig").OpTag;
const SliceRange = @import("../types/operation/options.zig").SliceRange;

pub const Saved = struct {
    inputs: []const *const Value = &.{},
    output: ?*const Value = null,
    aux: ?*const Value = null,
};

pub const Parent = struct {
    value: *Value,
    input_slot: usize,
};

pub const Node = struct {
    op_tag: OpTag,
    parents: []Parent,
    saved: Saved = .{},
    axis: ?usize = null,
    keepdim: ?bool = null,
    slice_ranges: ?[]const SliceRange = null,
    permute_axes: ?[]const usize = null,
    scalar_a: ?f64 = null,
    scalar_b: ?f64 = null,
};
