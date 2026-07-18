const Value = @import("../types/tensor/value.zig").Value;
const obs = @import("../../obs/index.zig");
const metrics = obs.metrics;
const execution_layout = @import("layout.zig");
const materialization_execution = @import("materialization.zig");
const prepared_execution = @import("prepared.zig");
const transfer_execution = @import("transfer.zig");

pub const Sink = struct {
    transfer_to_host_count: ?metrics.Id = null,
    transfer_to_host_bytes: ?metrics.Id = null,
    transfer_from_host_count: ?metrics.Id = null,
    transfer_from_host_bytes: ?metrics.Id = null,
    contiguity_fixup_count: ?metrics.Id = null,
    contiguity_fixup_bytes: ?metrics.Id = null,
};

pub fn recordTransferSummary(sink: Sink, summary: transfer_execution.TransferSummary) void {
    if (summary.to_host_count > 0) {
        add(sink.transfer_to_host_count, @as(i64, @intCast(summary.to_host_count)));
        add(sink.transfer_to_host_bytes, @as(i64, @intCast(summary.to_host_bytes)));
    }
    if (summary.from_host_count > 0) {
        add(sink.transfer_from_host_count, @as(i64, @intCast(summary.from_host_count)));
        add(sink.transfer_from_host_bytes, @as(i64, @intCast(summary.from_host_bytes)));
    }
}

pub fn recordMaterializationSummary(sink: Sink, summary: materialization_execution.MaterializationSummary) void {
    if (summary.contiguity_fixup_count > 0) {
        add(sink.contiguity_fixup_count, @as(i64, @intCast(summary.contiguity_fixup_count)));
        add(sink.contiguity_fixup_bytes, @as(i64, @intCast(summary.contiguity_fixup_bytes)));
    }
    recordTransferSummary(sink, summary.transfer);
}

pub fn recordPackedDenseMaterialization(sink: Sink, value: *const Value) void {
    const byte_len = value.shape.numel() * value.dtype.size();
    add(sink.contiguity_fixup_count, 1);
    add(sink.contiguity_fixup_bytes, @as(i64, @intCast(byte_len)));
    add(sink.transfer_to_host_count, 1);
    add(sink.transfer_to_host_bytes, @as(i64, @intCast(byte_len)));
    add(sink.transfer_from_host_count, 1);
    add(sink.transfer_from_host_bytes, @as(i64, @intCast(byte_len)));
}

pub fn prepareInputValue(
    sink: Sink,
    allocator: @import("std").mem.Allocator,
    value: *const Value,
    decision: execution_layout.InputLayoutDecision,
    source: @import("../types/tensor/storage.zig").Storage.Source,
) !prepared_execution.PreparedInputValue {
    const prepared = try prepared_execution.prepareInputValue(allocator, value, decision, source);
    if (prepared.materialized_packed_dense) recordPackedDenseMaterialization(sink, value);
    return prepared;
}

fn add(id: ?metrics.Id, delta: i64) void {
    if (id) |metric_id| metrics.add(metric_id, delta);
}
