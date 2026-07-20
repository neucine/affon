const Tensor = @import("../types/tensor/tensor.zig").Tensor;
const telemetry = @import("../telemetry.zig");
const execution_layout = @import("../plan/layout.zig");
const materialization_execution = @import("materialization.zig");
const prepared_execution = @import("prepared.zig");
const transfer_execution = @import("transfer.zig");

pub const Sink = struct {};

pub fn recordTransferSummary(sink: Sink, summary: transfer_execution.TransferSummary) void {
    _ = sink;
    if (summary.to_host_count > 0) {
        telemetry.add(telemetry.metrics.execution.transfer_to_host_count, @intCast(summary.to_host_count));
        telemetry.add(telemetry.metrics.execution.transfer_to_host_bytes, @intCast(summary.to_host_bytes));
    }
    if (summary.from_host_count > 0) {
        telemetry.add(telemetry.metrics.execution.transfer_from_host_count, @intCast(summary.from_host_count));
        telemetry.add(telemetry.metrics.execution.transfer_from_host_bytes, @intCast(summary.from_host_bytes));
    }
}

pub fn recordMaterializationSummary(sink: Sink, summary: materialization_execution.MaterializationSummary) void {
    if (summary.contiguity_fixup_count > 0) {
        telemetry.add(telemetry.metrics.execution.contiguity_fixup_count, @intCast(summary.contiguity_fixup_count));
        telemetry.add(telemetry.metrics.execution.contiguity_fixup_bytes, @intCast(summary.contiguity_fixup_bytes));
    }
    recordTransferSummary(sink, summary.transfer);
}

pub fn recordPackedDenseMaterialization(sink: Sink, value: *const Tensor) void {
    _ = sink;
    const byte_len = value.shape.numel() * value.dtype.size();
    telemetry.add(telemetry.metrics.execution.contiguity_fixup_count, 1);
    telemetry.add(telemetry.metrics.execution.contiguity_fixup_bytes, @intCast(byte_len));
    telemetry.add(telemetry.metrics.execution.transfer_to_host_count, 1);
    telemetry.add(telemetry.metrics.execution.transfer_to_host_bytes, @intCast(byte_len));
    telemetry.add(telemetry.metrics.execution.transfer_from_host_count, 1);
    telemetry.add(telemetry.metrics.execution.transfer_from_host_bytes, @intCast(byte_len));
}

pub fn prepareInputValue(
    sink: Sink,
    allocator: @import("std").mem.Allocator,
    value: *const Tensor,
    decision: execution_layout.InputLayoutDecision,
    source: @import("../types/tensor/storage.zig").Storage.Source,
) !prepared_execution.PreparedInputValue {
    const prepared = try prepared_execution.prepareInputValue(allocator, value, decision, source);
    if (prepared.materialized_packed_dense) recordPackedDenseMaterialization(sink, value);
    return prepared;
}
