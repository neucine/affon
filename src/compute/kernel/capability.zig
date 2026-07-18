const std = @import("std");
const Device = @import("../tensor/device.zig").Device;
const DType = @import("../tensor/dtype.zig").DType;
const Layout = @import("../tensor/layout.zig").Layout;
const Shape = @import("../tensor/shape.zig").Shape;

pub const OutputLayout = enum {
    dense,
    view,
    same_layout,
    backend_defined,
};

pub const KernelCapability = struct {
    accepts_dense: bool,
    accepts_offset: bool,
    accepts_strides: bool,
    accepts_broadcast_strides: bool,
    accepts_index_status: bool = false,
    requires_host: bool = false,
    output_layout: OutputLayout,
};

pub const InputLayoutDecision = enum {
    accept,
    pack_to_dense,
    unsupported,
};

pub fn unary(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByUnary(device, dtype));
}

pub fn unaryInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(unary(device, dtype), shape, layout);
}

pub fn binary(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByBinary(device, dtype));
}

pub fn binaryInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(binary(device, dtype), a_shape, a_layout, b_shape, b_layout);
}

pub fn binaryBroadcast(device: Device, dtype: DType) KernelCapability {
    return broadcastAware(dtypeSupportedByBinaryBroadcast(device, dtype));
}

pub fn binaryBroadcastInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(binaryBroadcast(device, dtype), a_shape, a_layout, b_shape, b_layout);
}

pub fn compare(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByCompare(device, dtype));
}

pub fn compareInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(compare(device, dtype), a_shape, a_layout, b_shape, b_layout);
}

pub fn compareBroadcast(device: Device, dtype: DType) KernelCapability {
    return broadcastAware(dtypeSupportedByCompareBroadcast(device, dtype));
}

pub fn compareBroadcastInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(compareBroadcast(device, dtype), a_shape, a_layout, b_shape, b_layout);
}

pub fn where(device: Device, cond_dtype: DType, value_dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByWhere(device, cond_dtype, value_dtype));
}

pub fn whereInputLayoutDecision(
    device: Device,
    cond_dtype: DType,
    value_dtype: DType,
    cond_shape: []const usize,
    cond_layout: Layout,
    on_true_shape: []const usize,
    on_true_layout: Layout,
    on_false_shape: []const usize,
    on_false_layout: Layout,
) InputLayoutDecision {
    return tripleInputLayoutDecision(
        where(device, cond_dtype, value_dtype),
        cond_shape,
        cond_layout,
        on_true_shape,
        on_true_layout,
        on_false_shape,
        on_false_layout,
    );
}

pub fn whereBroadcast(device: Device, cond_dtype: DType, value_dtype: DType) KernelCapability {
    return broadcastAware(dtypeSupportedByWhereBroadcast(device, cond_dtype, value_dtype));
}

pub fn whereBroadcastInputLayoutDecision(
    device: Device,
    cond_dtype: DType,
    value_dtype: DType,
    cond_shape: []const usize,
    cond_layout: Layout,
    on_true_shape: []const usize,
    on_true_layout: Layout,
    on_false_shape: []const usize,
    on_false_layout: Layout,
) InputLayoutDecision {
    return tripleInputLayoutDecision(
        whereBroadcast(device, cond_dtype, value_dtype),
        cond_shape,
        cond_layout,
        on_true_shape,
        on_true_layout,
        on_false_shape,
        on_false_layout,
    );
}

pub fn maskedFill(device: Device, input_dtype: DType, mask_dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByMaskedFill(device, input_dtype, mask_dtype));
}

pub fn maskedFillInputLayoutDecision(
    device: Device,
    input_dtype: DType,
    mask_dtype: DType,
    input_shape: []const usize,
    input_layout: Layout,
    mask_shape: []const usize,
    mask_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(
        maskedFill(device, input_dtype, mask_dtype),
        input_shape,
        input_layout,
        mask_shape,
        mask_layout,
    );
}

pub fn maskedFillBroadcast(device: Device, input_dtype: DType, mask_dtype: DType) KernelCapability {
    return broadcastAware(dtypeSupportedByMaskedFillBroadcast(device, input_dtype, mask_dtype));
}

pub fn maskedFillBroadcastInputLayoutDecision(
    device: Device,
    input_dtype: DType,
    mask_dtype: DType,
    input_shape: []const usize,
    input_layout: Layout,
    mask_shape: []const usize,
    mask_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(
        maskedFillBroadcast(device, input_dtype, mask_dtype),
        input_shape,
        input_layout,
        mask_shape,
        mask_layout,
    );
}

pub fn reductionAll(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByReductionAll(device, dtype));
}

pub fn reductionAllInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(reductionAll(device, dtype), shape, layout);
}

pub fn reductionAxis(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByReductionAxis(device, dtype));
}

pub fn reductionAxisInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(reductionAxis(device, dtype), shape, layout);
}

pub fn dot(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByDot(device, dtype));
}

pub fn dotInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(dot(device, dtype), a_shape, a_layout, b_shape, b_layout);
}

pub fn softmax(device: Device, dtype: DType) KernelCapability {
    return switch (device) {
        .cpu => denseOnly(dtypeSupportedBySoftmax(device, dtype)),
        .metal => .{
            .accepts_dense = dtypeSupportedBySoftmax(device, dtype),
            .accepts_offset = false,
            .accepts_strides = dtypeSupportedBySoftmax(device, dtype),
            .accepts_broadcast_strides = false,
            .output_layout = .dense,
        },
    };
}

pub fn softmaxInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    if (device == .metal and dtypeSupportedBySoftmax(device, dtype)) {
        if (shape.len > 8) return .unsupported;
        if (layout.strides.len != shape.len) return .pack_to_dense;
        if (isDenseLayout(shape, layout)) return .accept;
        if (layout.offset != 0) return .pack_to_dense;
        if (!stridesAreNonNegative(layout.strides)) return .pack_to_dense;
        return .accept;
    }
    return singleInputLayoutDecision(softmax(device, dtype), shape, layout);
}

pub fn logSoftmax(device: Device, dtype: DType) KernelCapability {
    return softmax(device, dtype);
}

pub fn logSoftmaxInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return softmaxInputLayoutDecision(device, dtype, shape, layout);
}

pub fn reduceToShape(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByReduceToShape(device, dtype));
}

pub fn reduceToShapeInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(reduceToShape(device, dtype), shape, layout);
}

pub fn logSoftmaxNll(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByLogSoftmaxNll(device, dtype));
}

pub fn logSoftmaxNllInputLayoutDecision(
    device: Device,
    dtype: DType,
    logits_shape: []const usize,
    logits_layout: Layout,
    targets_shape: []const usize,
    targets_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(logSoftmaxNll(device, dtype), logits_shape, logits_layout, targets_shape, targets_layout);
}

pub fn crossEntropyIndexed(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByCrossEntropyIndexed(device, dtype), true);
}

pub fn crossEntropyIndexedInputLayoutDecision(
    device: Device,
    dtype: DType,
    logits_shape: []const usize,
    logits_layout: Layout,
    targets_shape: []const usize,
    targets_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(crossEntropyIndexed(device, dtype), logits_shape, logits_layout, targets_shape, targets_layout);
}

pub fn crossEntropyIndexedBackward(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByCrossEntropyIndexedBackward(device, dtype), true);
}

pub fn crossEntropyIndexedBackwardInputLayoutDecision(
    device: Device,
    dtype: DType,
    logits_shape: []const usize,
    logits_layout: Layout,
    targets_shape: []const usize,
    targets_layout: Layout,
    grad_out_shape: []const usize,
    grad_out_layout: Layout,
) InputLayoutDecision {
    return tripleInputLayoutDecision(
        crossEntropyIndexedBackward(device, dtype),
        logits_shape,
        logits_layout,
        targets_shape,
        targets_layout,
        grad_out_shape,
        grad_out_layout,
    );
}

pub fn gather(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByGather(device, dtype), true);
}

pub fn gatherInputLayoutDecision(
    device: Device,
    dtype: DType,
    input_shape: []const usize,
    input_layout: Layout,
    index_shape: []const usize,
    index_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(gather(device, dtype), input_shape, input_layout, index_shape, index_layout);
}

pub fn embedding(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByEmbedding(device, dtype), true);
}

pub fn embeddingInputLayoutDecision(
    device: Device,
    dtype: DType,
    table_shape: []const usize,
    table_layout: Layout,
    index_shape: []const usize,
    index_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(embedding(device, dtype), table_shape, table_layout, index_shape, index_layout);
}

pub fn indexSelect(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByIndexSelect(device, dtype), true);
}

pub fn indexSelectInputLayoutDecision(
    device: Device,
    dtype: DType,
    input_shape: []const usize,
    input_layout: Layout,
    index_shape: []const usize,
    index_layout: Layout,
) InputLayoutDecision {
    return pairInputLayoutDecision(indexSelect(device, dtype), input_shape, input_layout, index_shape, index_layout);
}

pub fn scatterAdd(device: Device, dtype: DType) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByScatterAdd(device, dtype), true);
}

pub fn scatterAddInputLayoutDecision(
    device: Device,
    dtype: DType,
    base_shape: []const usize,
    base_layout: Layout,
    index_shape: []const usize,
    index_layout: Layout,
    updates_shape: []const usize,
    updates_layout: Layout,
) InputLayoutDecision {
    return tripleInputLayoutDecision(
        scatterAdd(device, dtype),
        base_shape,
        base_layout,
        index_shape,
        index_layout,
        updates_shape,
        updates_layout,
    );
}

pub fn oneHot(device: Device) KernelCapability {
    return denseOnlyWithIndexStatus(dtypeSupportedByOneHot(device), true);
}

pub fn oneHotInputLayoutDecision(device: Device, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(oneHot(device), shape, layout);
}

pub fn topK(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByTopK(device, dtype));
}

pub fn topKInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(topK(device, dtype), shape, layout);
}

pub fn cast(device: Device, from: DType, to: DType) KernelCapability {
    return denseOnly(dtypeSupportedByCast(device, from, to));
}

pub fn castInputLayoutDecision(device: Device, from: DType, to: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(cast(device, from, to), shape, layout);
}

pub fn layerNorm(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByLayerNorm(device, dtype));
}

pub fn layerNormInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(layerNorm(device, dtype), shape, layout);
}

pub fn rmsNorm(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByRmsNorm(device, dtype));
}

pub fn rmsNormInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(rmsNorm(device, dtype), shape, layout);
}

pub fn cat(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByCat(device, dtype));
}

pub fn catInputLayoutDecision(device: Device, dtype: DType, shapes: []const []const usize, layouts: []const Layout) InputLayoutDecision {
    return multiInputLayoutDecision(cat(device, dtype), shapes, layouts);
}

pub fn stack(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedByStack(device, dtype));
}

pub fn stackInputLayoutDecision(device: Device, dtype: DType, shapes: []const []const usize, layouts: []const Layout) InputLayoutDecision {
    return multiInputLayoutDecision(stack(device, dtype), shapes, layouts);
}

pub fn slice(device: Device, dtype: DType) KernelCapability {
    return denseOnly(dtypeSupportedBySlice(device, dtype));
}

pub fn sliceInputLayoutDecision(device: Device, dtype: DType, shape: []const usize, layout: Layout) InputLayoutDecision {
    return singleInputLayoutDecision(slice(device, dtype), shape, layout);
}

pub fn matmul(device: Device, dtype: DType) KernelCapability {
    const dtype_supported = switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };

    return .{
        .accepts_dense = dtype_supported,
        .accepts_offset = dtype_supported,
        .accepts_strides = dtype_supported,
        .accepts_broadcast_strides = dtype_supported,
        .output_layout = .dense,
    };
}

pub fn matmulAcceptsLayouts(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) bool {
    return matmulInputLayoutDecision(device, dtype, a_shape, a_layout, b_shape, b_layout) == .accept;
}

pub fn matmulInputLayoutDecision(
    device: Device,
    dtype: DType,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    const cap = matmul(device, dtype);
    if (cap.requires_host) return .unsupported;
    if (!cap.accepts_dense) return .unsupported;
    if (a_shape.len < 1 or a_shape.len > 8) return .unsupported;
    if (b_shape.len < 1 or b_shape.len > 8) return .unsupported;
    if (a_layout.strides.len != a_shape.len) return .pack_to_dense;
    if (b_layout.strides.len != b_shape.len) return .pack_to_dense;

    const a_dense = isDenseLayout(a_shape, a_layout);
    const b_dense = isDenseLayout(b_shape, b_layout);
    if (a_dense and b_dense) return .accept;

    if (!cap.accepts_offset and (a_layout.offset != 0 or b_layout.offset != 0)) return .pack_to_dense;
    if (!cap.accepts_strides) return .pack_to_dense;
    if (!stridesAreNonNegative(a_layout.strides)) return .pack_to_dense;
    if (!stridesAreNonNegative(b_layout.strides)) return .pack_to_dense;
    return .accept;
}

fn isDenseLayout(shape: []const usize, layout: Layout) bool {
    if (layout.offset != 0) return false;
    var expected: isize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        if (layout.strides[i] != expected) return false;
        expected *= @as(isize, @intCast(shape[i]));
    }
    return true;
}

fn stridesAreNonNegative(strides: []const isize) bool {
    for (strides) |stride| {
        if (stride < 0) return false;
    }
    return true;
}

fn denseOnly(dtype_supported: bool) KernelCapability {
    return .{
        .accepts_dense = dtype_supported,
        .accepts_offset = false,
        .accepts_strides = false,
        .accepts_broadcast_strides = false,
        .output_layout = .dense,
    };
}

fn denseOnlyWithIndexStatus(dtype_supported: bool, accepts_index_status: bool) KernelCapability {
    return .{
        .accepts_dense = dtype_supported,
        .accepts_offset = false,
        .accepts_strides = false,
        .accepts_broadcast_strides = false,
        .accepts_index_status = dtype_supported and accepts_index_status,
        .output_layout = .dense,
    };
}

fn broadcastAware(dtype_supported: bool) KernelCapability {
    return .{
        .accepts_dense = dtype_supported,
        .accepts_offset = dtype_supported,
        .accepts_strides = dtype_supported,
        .accepts_broadcast_strides = dtype_supported,
        .output_layout = .dense,
    };
}

fn singleInputLayoutDecision(cap: KernelCapability, shape: []const usize, layout: Layout) InputLayoutDecision {
    if (cap.requires_host) return .unsupported;
    if (!cap.accepts_dense) return .unsupported;
    if (shape.len > 8) return .unsupported;
    if (layout.strides.len != shape.len) return .pack_to_dense;
    if (isDenseLayout(shape, layout)) return .accept;
    if (!cap.accepts_offset and layout.offset != 0) return .pack_to_dense;
    if (!cap.accepts_strides) return .pack_to_dense;
    if (!stridesAreNonNegative(layout.strides)) return .pack_to_dense;
    return .accept;
}

fn pairInputLayoutDecision(
    cap: KernelCapability,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
) InputLayoutDecision {
    const a_decision = singleInputLayoutDecision(cap, a_shape, a_layout);
    if (a_decision != .accept) return a_decision;
    return singleInputLayoutDecision(cap, b_shape, b_layout);
}

fn tripleInputLayoutDecision(
    cap: KernelCapability,
    a_shape: []const usize,
    a_layout: Layout,
    b_shape: []const usize,
    b_layout: Layout,
    c_shape: []const usize,
    c_layout: Layout,
) InputLayoutDecision {
    const a_decision = singleInputLayoutDecision(cap, a_shape, a_layout);
    if (a_decision != .accept) return a_decision;
    const b_decision = singleInputLayoutDecision(cap, b_shape, b_layout);
    if (b_decision != .accept) return b_decision;
    return singleInputLayoutDecision(cap, c_shape, c_layout);
}

fn multiInputLayoutDecision(cap: KernelCapability, shapes: []const []const usize, layouts: []const Layout) InputLayoutDecision {
    if (shapes.len != layouts.len) return .unsupported;
    for (shapes, layouts) |shape, layout| {
        const decision = singleInputLayoutDecision(cap, shape, layout);
        if (decision != .accept) return decision;
    }
    return .accept;
}

fn dtypeSupportedByUnary(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

fn dtypeSupportedByBinary(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

fn dtypeSupportedByBinaryBroadcast(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByCompare(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

fn dtypeSupportedByCompareBroadcast(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByWhere(device: Device, cond_dtype: DType, value_dtype: DType) bool {
    return switch (device) {
        .cpu => switch (cond_dtype) {
            .f32, .f64, .i64 => switch (value_dtype) {
                .f32, .f64, .i64 => true,
            },
        },
        .metal => switch (value_dtype) {
            .f32, .i64 => cond_dtype == .f32 or cond_dtype == .i64,
            else => false,
        },
    };
}

fn dtypeSupportedByWhereBroadcast(device: Device, cond_dtype: DType, value_dtype: DType) bool {
    return switch (device) {
        .cpu => dtypeSupportedByWhere(device, cond_dtype, value_dtype),
        .metal => value_dtype == .f32 and (cond_dtype == .f32 or cond_dtype == .i64),
    };
}

fn dtypeSupportedByMaskedFill(device: Device, input_dtype: DType, mask_dtype: DType) bool {
    return switch (device) {
        .cpu => switch (mask_dtype) {
            .f32, .f64, .i64 => switch (input_dtype) {
                .f32, .f64, .i64 => true,
            },
        },
        .metal => mask_dtype == .i64 and (input_dtype == .f32 or input_dtype == .i64),
    };
}

fn dtypeSupportedByMaskedFillBroadcast(device: Device, input_dtype: DType, mask_dtype: DType) bool {
    return switch (device) {
        .cpu => dtypeSupportedByMaskedFill(device, input_dtype, mask_dtype),
        .metal => mask_dtype == .i64 and (input_dtype == .f32 or input_dtype == .i64),
    };
}

fn dtypeSupportedByReductionAll(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => switch (dtype) {
            .f32, .i64 => true,
            else => false,
        },
    };
}

fn dtypeSupportedByReductionAxis(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => switch (dtype) {
            .f32, .i64 => true,
            else => false,
        },
    };
}

fn dtypeSupportedByDot(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

fn dtypeSupportedBySoftmax(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => dtype == .f32 or dtype == .f64,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByLogSoftmax(device: Device, dtype: DType) bool {
    return dtypeSupportedBySoftmax(device, dtype);
}

fn dtypeSupportedByReduceToShape(device: Device, dtype: DType) bool {
    return dtypeSupportedByReductionAxis(device, dtype);
}

fn dtypeSupportedByLogSoftmaxNll(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => dtype == .f32 or dtype == .f64,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByCrossEntropyIndexed(device: Device, dtype: DType) bool {
    return dtypeSupportedByLogSoftmaxNll(device, dtype);
}

fn dtypeSupportedByCrossEntropyIndexedBackward(device: Device, dtype: DType) bool {
    return dtypeSupportedByLogSoftmaxNll(device, dtype);
}

fn dtypeSupportedByGather(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

fn dtypeSupportedByEmbedding(device: Device, dtype: DType) bool {
    return dtypeSupportedByGather(device, dtype);
}

fn dtypeSupportedByIndexSelect(device: Device, dtype: DType) bool {
    return dtypeSupportedByGather(device, dtype);
}

fn dtypeSupportedByScatterAdd(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByOneHot(device: Device) bool {
    return switch (device) {
        .cpu, .metal => true,
    };
}

fn dtypeSupportedByTopK(device: Device, dtype: DType) bool {
    return dtypeSupportedByGather(device, dtype);
}

fn dtypeSupportedByCast(device: Device, from: DType, to: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => switch (from) {
            .f32 => to == .f32 or to == .i64,
            .i64 => to == .i64 or to == .f32,
            .f64 => false,
        },
    };
}

fn dtypeSupportedByLayerNorm(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => dtype == .f32 or dtype == .f64,
        .metal => dtype == .f32,
    };
}

fn dtypeSupportedByRmsNorm(device: Device, dtype: DType) bool {
    return dtypeSupportedByLayerNorm(device, dtype);
}

fn dtypeSupportedByCat(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu, .metal => switch (dtype) {
            .f32, .f64, .i64 => true,
        },
    };
}

fn dtypeSupportedByStack(device: Device, dtype: DType) bool {
    return dtypeSupportedByCat(device, dtype);
}

fn dtypeSupportedBySlice(device: Device, dtype: DType) bool {
    return switch (device) {
        .cpu => true,
        .metal => dtype == .f32 or dtype == .i64,
    };
}

test "matmul capability accepts dense cpu layout" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ 3, 1 }, 0);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{ 4, 1 }, 0);
    defer rhs.deinit();

    try std.testing.expect(matmulAcceptsLayouts(.cpu, .f32, &.{ 2, 3 }, lhs, &.{ 3, 4 }, rhs));
}

test "matmul capability accepts transposed cpu layout" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{ 4, 1 }, 0);
    defer rhs.deinit();

    try std.testing.expect(matmulAcceptsLayouts(.cpu, .f32, &.{ 3, 2 }, lhs, &.{ 2, 4 }, rhs));
    try std.testing.expectEqual(
        InputLayoutDecision.accept,
        matmulInputLayoutDecision(.cpu, .f32, &.{ 3, 2 }, lhs, &.{ 2, 4 }, rhs),
    );
}

test "matmul capability rejects unsupported metal dtype" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ 3, 1 }, 0);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{ 4, 1 }, 0);
    defer rhs.deinit();

    try std.testing.expect(!matmulAcceptsLayouts(.metal, .f64, &.{ 2, 3 }, lhs, &.{ 3, 4 }, rhs));
    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        matmulInputLayoutDecision(.metal, .f64, &.{ 2, 3 }, lhs, &.{ 3, 4 }, rhs),
    );
}

test "matmul capability requests dense pack for negative strides" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ -1, 3 }, 2);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{ 4, 1 }, 0);
    defer rhs.deinit();

    try std.testing.expect(!matmulAcceptsLayouts(.cpu, .f32, &.{ 3, 2 }, lhs, &.{ 2, 4 }, rhs));
    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        matmulInputLayoutDecision(.cpu, .f32, &.{ 3, 2 }, lhs, &.{ 2, 4 }, rhs),
    );
}

test "binary broadcast capability accepts positive-stride cpu views" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ 3, 1 }, 1);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{1}, 0);
    defer rhs.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.accept,
        binaryBroadcastInputLayoutDecision(.cpu, .f64, &.{ 2, 3 }, lhs, &.{3}, rhs),
    );
}

test "binary capability requests dense pack for non-dense equal-shape input" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer lhs.deinit();
    var rhs_shape = try Shape.initCopy(allocator, &.{ 3, 2 });
    defer rhs_shape.deinit();
    var rhs = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        binaryInputLayoutDecision(.cpu, .f32, &.{ 3, 2 }, lhs, &.{ 3, 2 }, rhs),
    );
}

test "where broadcast capability rejects unsupported metal value dtype" {
    const allocator = std.testing.allocator;
    var cond_shape = try Shape.initCopy(allocator, &.{ 2, 1 });
    defer cond_shape.deinit();
    var cond = try Layout.initContiguous(allocator, cond_shape);
    defer cond.deinit();
    var on_true_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer on_true_shape.deinit();
    var on_true = try Layout.initContiguous(allocator, on_true_shape);
    defer on_true.deinit();
    var on_false_shape = try Shape.initCopy(allocator, &.{3});
    defer on_false_shape.deinit();
    var on_false = try Layout.initContiguous(allocator, on_false_shape);
    defer on_false.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        whereBroadcastInputLayoutDecision(.metal, .i64, .i64, &.{ 2, 1 }, cond, &.{ 2, 3 }, on_true, &.{3}, on_false),
    );
}

test "masked_fill broadcast capability requests dense pack for negative strides" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer input.deinit();
    var mask_shape = try Shape.initCopy(allocator, &.{ 3, 1 });
    defer mask_shape.deinit();
    var mask = try Layout.initContiguous(allocator, mask_shape);
    defer mask.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        maskedFillBroadcastInputLayoutDecision(.cpu, .f32, .i64, &.{ 3, 2 }, input, &.{ 3, 1 }, mask),
    );
}

test "reduction axis capability requests dense pack for transposed input" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        reductionAxisInputLayoutDecision(.cpu, .f32, &.{ 3, 2 }, input),
    );
}

test "softmax capability rejects unsupported metal dtype" {
    const allocator = std.testing.allocator;
    var input_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer input_shape.deinit();
    var input = try Layout.initContiguous(allocator, input_shape);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        softmaxInputLayoutDecision(.metal, .f64, &.{ 2, 3 }, input),
    );
}

test "softmax capability accepts offset-free positive-stride metal view" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{ 1, 2 }, 0);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.accept,
        softmaxInputLayoutDecision(.metal, .f32, &.{ 2, 2 }, input),
    );
    try std.testing.expectEqual(
        InputLayoutDecision.accept,
        logSoftmaxInputLayoutDecision(.metal, .f32, &.{ 2, 2 }, input),
    );
}

test "dot capability requests dense pack for negative strides" {
    const allocator = std.testing.allocator;
    var lhs = try Layout.initCopy(allocator, &.{-1}, 2);
    defer lhs.deinit();
    var rhs_shape = try Shape.initCopy(allocator, &.{3});
    defer rhs_shape.deinit();
    var rhs = try Layout.initContiguous(allocator, rhs_shape);
    defer rhs.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        dotInputLayoutDecision(.cpu, .f32, &.{3}, lhs, &.{3}, rhs),
    );
}

test "log_softmax_nll capability requests dense pack for non-dense input" {
    const allocator = std.testing.allocator;
    var logits = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer logits.deinit();
    var targets_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer targets_shape.deinit();
    var targets = try Layout.initContiguous(allocator, targets_shape);
    defer targets.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        logSoftmaxNllInputLayoutDecision(.cpu, .f32, &.{ 2, 2 }, logits, &.{ 2, 2 }, targets),
    );
}

test "gather capability requests dense pack for non-dense input" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{ 1, 3 }, 0);
    defer input.deinit();
    var index_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer index_shape.deinit();
    var index = try Layout.initContiguous(allocator, index_shape);
    defer index.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        gatherInputLayoutDecision(.cpu, .f32, &.{ 2, 2 }, input, &.{ 2, 2 }, index),
    );
}

test "indexed capabilities declare internal index status handling" {
    try std.testing.expect(gather(.cpu, .f32).accepts_index_status);
    try std.testing.expect(gather(.metal, .f32).accepts_index_status);
    try std.testing.expect(embedding(.cpu, .f32).accepts_index_status);
    try std.testing.expect(indexSelect(.metal, .f32).accepts_index_status);
    try std.testing.expect(scatterAdd(.cpu, .f32).accepts_index_status);
    try std.testing.expect(oneHot(.metal).accepts_index_status);
    try std.testing.expect(crossEntropyIndexed(.cpu, .f32).accepts_index_status);
    try std.testing.expect(crossEntropyIndexedBackward(.metal, .f32).accepts_index_status);
}

test "non-index capability does not claim index status handling" {
    try std.testing.expect(!topK(.metal, .f32).accepts_index_status);
    try std.testing.expect(!matmul(.cpu, .f32).accepts_index_status);
}

test "scatter_add capability rejects unsupported metal dtype" {
    const allocator = std.testing.allocator;
    var base_shape = try Shape.initCopy(allocator, &.{ 2, 3 });
    defer base_shape.deinit();
    var base = try Layout.initContiguous(allocator, base_shape);
    defer base.deinit();
    var index = try Layout.initContiguous(allocator, base_shape);
    defer index.deinit();
    var updates = try Layout.initContiguous(allocator, base_shape);
    defer updates.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        scatterAddInputLayoutDecision(.metal, .i64, &.{ 2, 3 }, base, &.{ 2, 3 }, index, &.{ 2, 3 }, updates),
    );
}

test "topk capability requests dense pack for negative strides" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{-1}, 4);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        topKInputLayoutDecision(.cpu, .f32, &.{5}, input),
    );
}

test "cast capability rejects unsupported metal f64 path" {
    const allocator = std.testing.allocator;
    var input_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer input_shape.deinit();
    var input = try Layout.initContiguous(allocator, input_shape);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        castInputLayoutDecision(.metal, .f64, .f32, &.{ 2, 2 }, input),
    );
}

test "layer_norm capability requests dense pack for negative strides" {
    const allocator = std.testing.allocator;
    var input = try Layout.initCopy(allocator, &.{ -2, 1 }, 4);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        layerNormInputLayoutDecision(.cpu, .f32, &.{ 3, 2 }, input),
    );
}

test "cat capability requests dense pack when one input is non-dense" {
    const allocator = std.testing.allocator;
    var lhs_shape = try Shape.initCopy(allocator, &.{ 2, 2 });
    defer lhs_shape.deinit();
    var lhs = try Layout.initContiguous(allocator, lhs_shape);
    defer lhs.deinit();
    var rhs = try Layout.initCopy(allocator, &.{ -2, 1 }, 2);
    defer rhs.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.pack_to_dense,
        catInputLayoutDecision(.cpu, .f32, &.{ &.{ 2, 2 }, &.{ 2, 1 } }, &.{ lhs, rhs }),
    );
}

test "slice capability rejects unsupported metal dtype" {
    const allocator = std.testing.allocator;
    var input_shape = try Shape.initCopy(allocator, &.{6});
    defer input_shape.deinit();
    var input = try Layout.initContiguous(allocator, input_shape);
    defer input.deinit();

    try std.testing.expectEqual(
        InputLayoutDecision.unsupported,
        sliceInputLayoutDecision(.metal, .f64, &.{6}, input),
    );
}
