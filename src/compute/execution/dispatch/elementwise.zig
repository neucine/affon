const Device = @import("../../shared/types/tensor/device.zig").Device;
const DType = @import("../../shared/types/tensor/dtype.zig").DType;
const Storage = @import("../../shared/types/tensor/storage.zig").Storage;
const Tensor = @import("../../shared/types/tensor/tensor.zig").Tensor;
const OpTag = @import("../../shared/types/operation/tag.zig").OpTag;
const kernel_dispatch = @import("../../backend/dispatch.zig");
const prepared_execution = @import("../prepared.zig");

pub fn dispatchUnary(
    device: Device,
    tag: OpTag,
    dtype: DType,
    input: *const Tensor,
    output: *Storage,
) !void {
    try kernel_dispatch.unary(
        device,
        tag,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
    );
}

pub fn dispatchClamp(
    device: Device,
    dtype: DType,
    input: *const Tensor,
    output: *Storage,
    min: f64,
    max: f64,
) !void {
    try kernel_dispatch.unaryClamp(
        device,
        dtype,
        input.storage orelse return error.InputNotMaterialized,
        output,
        min,
        max,
    );
}

pub fn dispatchBinary(
    device: Device,
    tag: OpTag,
    dtype: DType,
    lhs: *const Tensor,
    rhs: *const Tensor,
    output: *Storage,
    descriptor: prepared_execution.BinaryElementwiseDescriptor,
) !void {
    switch (descriptor) {
        .dense => try kernel_dispatch.binary(
            device,
            tag,
            dtype,
            lhs.storage orelse return error.InputNotMaterialized,
            rhs.storage orelse return error.InputNotMaterialized,
            output,
        ),
        .broadcast => |lowered| try kernel_dispatch.binaryBroadcast(
            device,
            tag,
            dtype,
            lhs.storage orelse return error.InputNotMaterialized,
            rhs.storage orelse return error.InputNotMaterialized,
            output,
            lowered.shape[0..lowered.rank],
            lowered.lhs_strides[0..lowered.rank],
            lowered.rhs_strides[0..lowered.rank],
            lhs.layout.offset,
            rhs.layout.offset,
        ),
    }
}

pub fn dispatchCompare(
    device: Device,
    tag: OpTag,
    input_dtype: DType,
    lhs: *const Tensor,
    rhs: *const Tensor,
    output: *Storage,
    descriptor: prepared_execution.BinaryElementwiseDescriptor,
) !void {
    switch (descriptor) {
        .dense => try kernel_dispatch.compare(
            device,
            tag,
            input_dtype,
            lhs.storage orelse return error.InputNotMaterialized,
            rhs.storage orelse return error.InputNotMaterialized,
            output,
        ),
        .broadcast => |lowered| try kernel_dispatch.compareBroadcast(
            device,
            tag,
            input_dtype,
            lhs.storage orelse return error.InputNotMaterialized,
            rhs.storage orelse return error.InputNotMaterialized,
            output,
            lowered.shape[0..lowered.rank],
            lowered.lhs_strides[0..lowered.rank],
            lowered.rhs_strides[0..lowered.rank],
            lhs.layout.offset,
            rhs.layout.offset,
        ),
    }
}

pub fn dispatchWhere(
    device: Device,
    cond_dtype: DType,
    value_dtype: DType,
    cond: *const Tensor,
    on_true: *const Tensor,
    on_false: *const Tensor,
    output: *Storage,
    descriptor: prepared_execution.WhereDescriptor,
) !void {
    switch (descriptor) {
        .dense => try kernel_dispatch.where(
            device,
            cond_dtype,
            value_dtype,
            cond.storage orelse return error.InputNotMaterialized,
            on_true.storage orelse return error.InputNotMaterialized,
            on_false.storage orelse return error.InputNotMaterialized,
            output,
        ),
        .broadcast => |lowered| try kernel_dispatch.whereBroadcast(
            device,
            cond_dtype,
            value_dtype,
            cond.storage orelse return error.InputNotMaterialized,
            on_true.storage orelse return error.InputNotMaterialized,
            on_false.storage orelse return error.InputNotMaterialized,
            output,
            lowered.shape[0..lowered.rank],
            lowered.cond_strides[0..lowered.rank],
            lowered.on_true_strides[0..lowered.rank],
            lowered.on_false_strides[0..lowered.rank],
            cond.layout.offset,
            on_true.layout.offset,
            on_false.layout.offset,
        ),
    }
}

pub fn dispatchMaskedFill(
    device: Device,
    input_dtype: DType,
    mask_dtype: DType,
    input: *const Tensor,
    mask: *const Tensor,
    output: *Storage,
    value: f64,
    descriptor: prepared_execution.MaskedFillDescriptor,
) !void {
    switch (descriptor) {
        .dense => try kernel_dispatch.maskedFill(
            device,
            input_dtype,
            mask_dtype,
            input.storage orelse return error.InputNotMaterialized,
            mask.storage orelse return error.InputNotMaterialized,
            output,
            value,
        ),
        .broadcast => |lowered| try kernel_dispatch.maskedFillBroadcast(
            device,
            input_dtype,
            mask_dtype,
            input.storage orelse return error.InputNotMaterialized,
            mask.storage orelse return error.InputNotMaterialized,
            output,
            lowered.shape[0..lowered.rank],
            lowered.input_strides[0..lowered.rank],
            lowered.mask_strides[0..lowered.rank],
            input.layout.offset,
            mask.layout.offset,
            value,
        ),
    }
}
