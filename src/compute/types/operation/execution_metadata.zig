const matmul_planning = @import("../../plan/matmul.zig");

pub const ExecutionMetadata = struct {
    matmul_hint: matmul_planning.MatmulHint = .none,
    hint_source: matmul_planning.HintSource = .none,

    pub fn none() ExecutionMetadata {
        return .{};
    }
};
