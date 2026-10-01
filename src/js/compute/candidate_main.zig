const std = @import("std");
const candidate = @import("candidate_core");
const candidate_onnx = @import("candidate_onnx");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    // std:process supplies the child arguments without argv[0], while a direct
    // invocation includes it. Accept both forms so the audit and CLI share the
    // exact same executable.
    const offset: usize = if (args.len > 0 and (std.mem.eql(u8, args[0], "onnx") or std.mem.eql(u8, args[0], "cpu") or std.mem.eql(u8, args[0], "metal") or std.mem.eql(u8, args[0], "cuda"))) 0 else 1;
    if (args.len > offset + 1 and std.mem.eql(u8, args[offset], "onnx")) {
        const case_index = if (args.len > offset + 2) try std.fmt.parseInt(usize, args[offset + 2], 10) else 0;
        var result = try candidate_onnx.run(allocator, init.io, args[offset + 1], .cpu, case_index);
        defer result.deinit(allocator);
        var imported_output: std.Io.Writer.Allocating = .init(allocator);
        defer imported_output.deinit();
        try imported_output.writer.print("{f}\n", .{std.json.fmt(.{
            .case = case_index,
            .onnx = result.onnx,
            .pytorch = result.pytorch,
            .top1 = result.top1,
            .values = result.values,
            .explanation = result.explanation,
            .passed = result.onnx.passed and result.pytorch.passed,
        }, .{})});
        try std.Io.File.stdout().writeStreamingAll(init.io, imported_output.written());
        return;
    }
    const backend_name = if (args.len > offset) args[offset] else "cpu";
    const backend: candidate.Backend = if (std.mem.eql(u8, backend_name, "cpu")) .cpu else if (std.mem.eql(u8, backend_name, "metal")) .metal else if (std.mem.eql(u8, backend_name, "cuda")) .cuda else return error.InvalidBackend;
    if (!candidate.backendAvailable(backend)) return error.BackendUnavailable;
    var proof = try candidate.runDecoderProof(allocator, backend);
    defer proof.deinit(allocator);
    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try output.writer.print("{f}\n", .{std.json.fmt(.{
        .backend = backend_name,
        .loss = proof.loss,
        .resumedParameter = proof.resumed_parameter,
        .explanation = proof.explanation,
    }, .{})});
    try std.Io.File.stdout().writeStreamingAll(init.io, output.written());
}
