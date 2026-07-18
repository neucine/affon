const semantic = @import("sema/index.zig");

pub const ExecutionKind = enum {
    elementwise_binary,
    elementwise_unary,
    elementwise_generic,
    reduction_all,
    reduction,
    index,
    view,
};

pub const AllocationIntent = enum {
    new_storage,
    view_only,
};

pub const InputRequirement = enum {
    preserve,
    require_storage,
    require_contiguous_input,
};

pub fn executionKindFromSemantic(kind: semantic.ExecutionKind) ExecutionKind {
    return switch (kind) {
        .elementwise_binary => .elementwise_binary,
        .elementwise_unary => .elementwise_unary,
        .elementwise_generic => .elementwise_generic,
        .reduction_all => .reduction_all,
        .reduction => .reduction,
        .index => .index,
        .view => .view,
    };
}

pub fn executionKindToSemantic(kind: ExecutionKind) semantic.ExecutionKind {
    return switch (kind) {
        .elementwise_binary => .elementwise_binary,
        .elementwise_unary => .elementwise_unary,
        .elementwise_generic => .elementwise_generic,
        .reduction_all => .reduction_all,
        .reduction => .reduction,
        .index => .index,
        .view => .view,
    };
}

pub fn allocationIntentFromSemantic(allocation: semantic.AllocationIntent) AllocationIntent {
    return switch (allocation) {
        .new_storage => .new_storage,
        .view_only => .view_only,
    };
}

pub fn allocationIntentToSemantic(allocation: AllocationIntent) semantic.AllocationIntent {
    return switch (allocation) {
        .new_storage => .new_storage,
        .view_only => .view_only,
    };
}

pub fn inputRequirementFromSemantic(requirement: semantic.InputRequirement) InputRequirement {
    return switch (requirement) {
        .preserve => .preserve,
        .require_storage => .require_storage,
        .require_contiguous_input => .require_contiguous_input,
    };
}

pub fn inputRequirementToSemantic(requirement: InputRequirement) semantic.InputRequirement {
    return switch (requirement) {
        .preserve => .preserve,
        .require_storage => .require_storage,
        .require_contiguous_input => .require_contiguous_input,
    };
}
