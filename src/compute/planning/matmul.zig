const std = @import("std");
const tensor = @import("../tensor/index.zig");
const ValueSpec = tensor.ValueSpec;
const DType = tensor.DType;
const Device = tensor.Device;
const Shape = tensor.Shape;
const Layout = tensor.Layout;

// This module owns structural/workload interpretation for matmul planning.
// It consumes already-validated math specs and must not re-derive infer-layer
// legality rules beyond what is required to describe axes and families.

pub const MatmulExecutionFamily = enum {
    gemm_2d,
    gemm_batched,
    gemm_projection,
    gemm_attention_scores,
    gemm_attention_values,
    gemm_generic_unresolved,
};

pub const ClassificationConfidence = enum {
    high,
    medium,
    low,
};

pub const MatmulHint = enum {
    none,
    projection,
    attention_scores,
    attention_values,
};

pub const HintSource = enum {
    none,
    graph_context,
    higher_level_module,
    api_execution_arg,
};

pub const MatmulClassificationContext = struct {
    hint: MatmulHint = .none,
    hint_source: HintSource = .none,
};

pub const AxisBinding = struct {
    axis: ?usize = null,
};

pub const MatmulDescriptor = struct {
    family: MatmulExecutionFamily,
    confidence: ClassificationConfidence,
    hint: MatmulHint,
    hint_source: HintSource,
    lhs_rank: usize,
    rhs_rank: usize,
    out_rank: usize,
    batch_rank: usize,
    lhs_row: AxisBinding,
    lhs_contraction: AxisBinding,
    rhs_contraction: AxisBinding,
    rhs_col: AxisBinding,
    out_row: AxisBinding,
    out_col: AxisBinding,
    flattenable_leading_batch: bool,

    pub fn hinted(self: MatmulDescriptor) bool {
        return self.hint != .none;
    }
};

pub fn familyName(family: MatmulExecutionFamily) []const u8 {
    return switch (family) {
        .gemm_2d => "gemm_2d",
        .gemm_batched => "gemm_batched",
        .gemm_projection => "gemm_projection",
        .gemm_attention_scores => "gemm_attention_scores",
        .gemm_attention_values => "gemm_attention_values",
        .gemm_generic_unresolved => "gemm_generic_unresolved",
    };
}

pub fn classifyFromSpecs(
    lhs: ValueSpec,
    rhs: ValueSpec,
    out: ValueSpec,
    ctx: MatmulClassificationContext,
) MatmulDescriptor {
    const lhs_rank = lhs.shape.rank();
    const rhs_rank = rhs.shape.rank();
    const out_rank = out.shape.rank();
    const batch_rank = batchRank(lhs_rank, rhs_rank, out_rank);
    const base = MatmulDescriptor{
        .family = .gemm_generic_unresolved,
        .confidence = .low,
        .hint = ctx.hint,
        .hint_source = ctx.hint_source,
        .lhs_rank = lhs_rank,
        .rhs_rank = rhs_rank,
        .out_rank = out_rank,
        .batch_rank = batch_rank,
        .lhs_row = .{ .axis = lhsRowAxis(lhs_rank) },
        .lhs_contraction = .{ .axis = lhsContractionAxis(lhs_rank) },
        .rhs_contraction = .{ .axis = rhsContractionAxis(rhs_rank) },
        .rhs_col = .{ .axis = rhsColAxis(rhs_rank) },
        .out_row = .{ .axis = outRowAxis(lhs_rank, rhs_rank, out_rank) },
        .out_col = .{ .axis = outColAxis(lhs_rank, rhs_rank, out_rank) },
        .flattenable_leading_batch = flattenableLeadingBatch(lhs_rank, rhs_rank),
    };

    if (ctx.hint != .none) return applyHint(base, ctx.hint);
    if (lhs_rank == 2 and rhs_rank == 2 and out_rank == 2) {
        return withFamily(base, .gemm_2d, .high);
    }
    if (lhs_rank >= 3 and rhs_rank == 2 and out_rank == lhs_rank) {
        return withFamily(base, .gemm_projection, .medium);
    }
    if (lhs_rank >= 3 and rhs_rank >= 3 and out_rank >= 2) {
        return withFamily(base, .gemm_batched, .medium);
    }
    return base;
}

fn applyHint(base: MatmulDescriptor, hint: MatmulHint) MatmulDescriptor {
    return switch (hint) {
        .none => base,
        .projection => withFamily(base, .gemm_projection, .high),
        .attention_scores => withFamily(base, .gemm_attention_scores, .high),
        .attention_values => withFamily(base, .gemm_attention_values, .high),
    };
}

fn withFamily(
    base: MatmulDescriptor,
    family: MatmulExecutionFamily,
    confidence: ClassificationConfidence,
) MatmulDescriptor {
    var out = base;
    out.family = family;
    out.confidence = confidence;
    return out;
}

fn batchRank(lhs_rank: usize, rhs_rank: usize, out_rank: usize) usize {
    if (lhs_rank == 1 and rhs_rank == 1) return 0;
    if (out_rank < 2) return 0;
    return out_rank - 2;
}

fn lhsRowAxis(lhs_rank: usize) ?usize {
    return if (lhs_rank == 1) null else lhs_rank - 2;
}

fn lhsContractionAxis(lhs_rank: usize) ?usize {
    return lhs_rank - 1;
}

fn rhsContractionAxis(rhs_rank: usize) ?usize {
    return if (rhs_rank == 1) 0 else rhs_rank - 2;
}

fn rhsColAxis(rhs_rank: usize) ?usize {
    return if (rhs_rank == 1) null else rhs_rank - 1;
}

fn outRowAxis(lhs_rank: usize, rhs_rank: usize, out_rank: usize) ?usize {
    if (lhs_rank == 1 and rhs_rank == 1) return null;
    if (lhs_rank == 1) return null;
    if (out_rank < 2) return null;
    return out_rank - 2;
}

fn outColAxis(lhs_rank: usize, rhs_rank: usize, out_rank: usize) ?usize {
    if (lhs_rank == 1 and rhs_rank == 1) return null;
    if (rhs_rank == 1) return null;
    if (out_rank == 0) return null;
    return out_rank - 1;
}

fn flattenableLeadingBatch(lhs_rank: usize, rhs_rank: usize) bool {
    return lhs_rank >= 3 and rhs_rank == 2;
}

fn makeValueSpec(allocator: std.mem.Allocator, dims: []const usize) !ValueSpec {
    var shape = try Shape.initCopy(allocator, dims);
    errdefer shape.deinit();
    var layout = try Layout.initContiguous(allocator, shape);
    errdefer layout.deinit();
    return .{
        .shape = shape,
        .dtype = .f32,
        .layout = layout,
        .device = .cpu,
    };
}

fn deinitValueSpec(spec: *ValueSpec) void {
    spec.layout.deinit();
    spec.shape.deinit();
    spec.* = undefined;
}

test "matmul semantics classifies plain 2d gemm" {
    const allocator = std.testing.allocator;
    var lhs = try makeValueSpec(allocator, &.{ 128, 64 });
    defer deinitValueSpec(&lhs);
    var rhs = try makeValueSpec(allocator, &.{ 64, 256 });
    defer deinitValueSpec(&rhs);
    var out = try makeValueSpec(allocator, &.{ 128, 256 });
    defer deinitValueSpec(&out);

    const descriptor = classifyFromSpecs(lhs, rhs, out, .{});
    try std.testing.expectEqual(MatmulExecutionFamily.gemm_2d, descriptor.family);
    try std.testing.expectEqual(ClassificationConfidence.high, descriptor.confidence);
    try std.testing.expectEqual(@as(usize, 0), descriptor.batch_rank);
    try std.testing.expectEqual(@as(?usize, 0), descriptor.lhs_row.axis);
    try std.testing.expectEqual(@as(?usize, 1), descriptor.lhs_contraction.axis);
    try std.testing.expectEqual(@as(?usize, 1), descriptor.out_col.axis);
    try std.testing.expect(!descriptor.hinted());
}

test "matmul semantics classifies projection conservatively" {
    const allocator = std.testing.allocator;
    var lhs = try makeValueSpec(allocator, &.{ 4, 256, 512 });
    defer deinitValueSpec(&lhs);
    var rhs = try makeValueSpec(allocator, &.{ 512, 2048 });
    defer deinitValueSpec(&rhs);
    var out = try makeValueSpec(allocator, &.{ 4, 256, 2048 });
    defer deinitValueSpec(&out);

    const descriptor = classifyFromSpecs(lhs, rhs, out, .{});
    try std.testing.expectEqual(MatmulExecutionFamily.gemm_projection, descriptor.family);
    try std.testing.expectEqual(ClassificationConfidence.medium, descriptor.confidence);
    try std.testing.expect(descriptor.flattenable_leading_batch);
    try std.testing.expectEqual(@as(usize, 1), descriptor.batch_rank);
}

test "matmul semantics uses explicit attention hint without guessing" {
    const allocator = std.testing.allocator;
    var lhs = try makeValueSpec(allocator, &.{ 4, 8, 256, 64 });
    defer deinitValueSpec(&lhs);
    var rhs = try makeValueSpec(allocator, &.{ 4, 8, 64, 256 });
    defer deinitValueSpec(&rhs);
    var out = try makeValueSpec(allocator, &.{ 4, 8, 256, 256 });
    defer deinitValueSpec(&out);

    const descriptor = classifyFromSpecs(lhs, rhs, out, .{
        .hint = .attention_scores,
        .hint_source = .higher_level_module,
    });
    try std.testing.expectEqual(MatmulExecutionFamily.gemm_attention_scores, descriptor.family);
    try std.testing.expectEqual(ClassificationConfidence.high, descriptor.confidence);
    try std.testing.expectEqual(MatmulHint.attention_scores, descriptor.hint);
    try std.testing.expectEqual(HintSource.higher_level_module, descriptor.hint_source);
    try std.testing.expect(descriptor.hinted());
}

test "matmul semantics keeps unhinted high-rank batched case generic to the workload family" {
    const allocator = std.testing.allocator;
    var lhs = try makeValueSpec(allocator, &.{ 4, 8, 256, 64 });
    defer deinitValueSpec(&lhs);
    var rhs = try makeValueSpec(allocator, &.{ 4, 8, 64, 256 });
    defer deinitValueSpec(&rhs);
    var out = try makeValueSpec(allocator, &.{ 4, 8, 256, 256 });
    defer deinitValueSpec(&out);

    const descriptor = classifyFromSpecs(lhs, rhs, out, .{});
    try std.testing.expectEqual(MatmulExecutionFamily.gemm_batched, descriptor.family);
    try std.testing.expectEqual(ClassificationConfidence.medium, descriptor.confidence);
    try std.testing.expectEqual(@as(usize, 2), descriptor.batch_rank);
}
