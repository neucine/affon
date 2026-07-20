const std = @import("std");

pub fn inverseAxes(allocator: std.mem.Allocator, axes: []const usize) ![]usize {
    const inv = try allocator.alloc(usize, axes.len);
    for (axes, 0..) |axis, i| inv[axis] = i;
    return inv;
}

fn transposeLastTwo(ctx: anytype, id: anytype) !@TypeOf(id) {
    const spec = ctx.valueSpec(id);
    const rank = spec.shape.dims.len;
    if (rank < 2) return error.GradUnsupported;
    const axes = try ctx.getAllocator().alloc(usize, rank);
    defer ctx.getAllocator().free(axes);
    for (0..rank) |i| axes[i] = i;
    const a = rank - 2;
    const b = rank - 1;
    const tmp = axes[a];
    axes[a] = axes[b];
    axes[b] = tmp;
    return ctx.permute(id, axes);
}

pub fn deriveParentGrad(ctx: anytype, grad_out: anytype, step: anytype, input_slot: usize) !@TypeOf(grad_out) {
    return switch (step.op_tag) {
        .add => ctx.reduceToParentShape(grad_out, step.saved.inputs[input_slot]),
        .sub => blk: {
            var id = try ctx.reduceToParentShape(grad_out, step.saved.inputs[input_slot]);
            if (input_slot == 1) id = try ctx.neg(id);
            break :blk id;
        },
        .mul => blk: {
            const other = try ctx.bindTensor(@constCast(step.saved.inputs[if (input_slot == 0) 1 else 0]));
            const mul = try ctx.mul(grad_out, other);
            break :blk try ctx.reduceToParentShape(mul, step.saved.inputs[input_slot]);
        },
        .div => blk: {
            const a = try ctx.bindTensor(@constCast(step.saved.inputs[0]));
            const b = try ctx.bindTensor(@constCast(step.saved.inputs[1]));
            if (input_slot == 0) {
                const div = try ctx.div(grad_out, b);
                break :blk try ctx.reduceToParentShape(div, step.saved.inputs[0]);
            }
            const bb = try ctx.mul(b, b);
            const ga = try ctx.mul(grad_out, a);
            const div = try ctx.div(ga, bb);
            const neg = try ctx.neg(div);
            break :blk try ctx.reduceToParentShape(neg, step.saved.inputs[1]);
        },
        .neg => ctx.neg(grad_out),
        .sign => ctx.fullLike(step.saved.inputs[0], 0.0),
        .abs => blk: {
            const x = try ctx.bindTensor(@constCast(step.saved.inputs[0]));
            const sx = try ctx.sign(x);
            break :blk try ctx.mul(grad_out, sx);
        },
        .sqrt => blk: {
            const y = try ctx.bindTensor(@constCast(step.saved.output orelse return error.GradUnsupported));
            const two = try ctx.fullLike(step.saved.inputs[0], 2.0);
            const denom_l = try ctx.mul(two, y);
            break :blk try ctx.div(grad_out, denom_l);
        },
        .exp => blk: {
            const out = try ctx.bindTensor(@constCast(step.saved.output orelse return error.GradUnsupported));
            break :blk try ctx.mul(grad_out, out);
        },
        .log => blk: {
            const input = try ctx.bindTensor(@constCast(step.saved.inputs[0]));
            break :blk try ctx.div(grad_out, input);
        },
        .sigmoid => blk: {
            const one = try ctx.scalarLike(step.saved.inputs[0], 1.0);
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const one_minus = try ctx.sub(one, y);
            const yy = try ctx.mul(y, one_minus);
            break :blk try ctx.mul(grad_out, yy);
        },
        .silu => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const sig = try ctx.sigmoid(x);
            const one = try ctx.scalarLike(step.saved.inputs[0], 1.0);
            const one_minus_sig = try ctx.sub(one, sig);
            const x_sig = try ctx.mul(x, sig);
            const x_sig_term = try ctx.mul(x_sig, one_minus_sig);
            const local = try ctx.add(sig, x_sig_term);
            break :blk try ctx.mul(grad_out, local);
        },
        .tanh => blk: {
            const one = try ctx.scalarLike(step.saved.inputs[0], 1.0);
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const yy = try ctx.mul(y, y);
            const one_minus = try ctx.sub(one, yy);
            break :blk try ctx.mul(grad_out, one_minus);
        },
        .relu => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const rx = try ctx.relu(x);
            const mask = try ctx.sign(rx);
            break :blk try ctx.mul(grad_out, mask);
        },
        .gelu => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const grad = try ctx.geluGrad(x);
            break :blk try ctx.mul(grad_out, grad);
        },
        .clamp => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const minv = step.scalar_a orelse return error.GradUnsupported;
            const maxv = step.scalar_b orelse return error.GradUnsupported;
            const zero = try ctx.fullLike(step.saved.inputs[0], 0.0);
            const min_tensor = try ctx.scalarLike(step.saved.inputs[0], minv);
            const max_tensor = try ctx.scalarLike(step.saved.inputs[0], maxv);
            const above_min = try ctx.gt(x, min_tensor);
            const below_max = try ctx.lt(x, max_tensor);
            const inner_grad = try ctx.whereSelect(above_min, grad_out, zero);
            break :blk try ctx.whereSelect(below_max, inner_grad, zero);
        },
        .reshape => blk: {
            if (ctx.sameShape(grad_out, step.saved.inputs[0].shape.dims)) break :blk try ctx.identity(grad_out);
            const dense = try ctx.contiguous(grad_out);
            break :blk try ctx.reshape(dense, step.saved.inputs[0].shape.dims);
        },
        .contiguous => blk: {
            if (ctx.sameShape(grad_out, step.saved.inputs[0].shape.dims)) break :blk try ctx.identity(grad_out);
            break :blk try ctx.reshape(grad_out, step.saved.inputs[0].shape.dims);
        },
        .transpose => ctx.transpose(grad_out),
        .permute => blk: {
            const axes = step.permute_axes orelse return error.GradUnsupported;
            const inv = try inverseAxes(ctx.getAllocator(), axes);
            defer ctx.getAllocator().free(inv);
            break :blk try ctx.permute(grad_out, inv);
        },
        .squeeze => try ctx.unsqueeze(grad_out, step.axis orelse return error.GradUnsupported),
        .unsqueeze => try ctx.squeeze(grad_out, step.axis orelse return error.GradUnsupported),
        .cast => try ctx.cast(grad_out, step.saved.inputs[0].dtype),
        .masked_fill => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const mask = try ctx.bindTensor(@constCast(step.saved.inputs[1]));
            break :blk try ctx.maskedFillZero(grad_out, mask);
        },
        .sum_all => blk: {
            const ones = try ctx.fullLike(step.saved.inputs[0], 1.0);
            break :blk try ctx.mul(ones, grad_out);
        },
        .mean_all => blk: {
            const n = @as(f64, @floatFromInt(step.saved.inputs[0].shape.numel()));
            const scale = try ctx.fullLike(step.saved.inputs[0], 1.0 / n);
            break :blk try ctx.mul(scale, grad_out);
        },
        .min_all, .max_all => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const mask = try ctx.eq(x, y);
            const zero = try ctx.fullLike(step.saved.inputs[0], 0.0);
            break :blk try ctx.whereSelect(mask, grad_out, zero);
        },
        .variance_all => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const mean = try ctx.meanAll(x);
            const centered = try ctx.sub(x, mean);
            const n = @as(f64, @floatFromInt(step.saved.inputs[0].shape.numel()));
            const scale = try ctx.scalarLike(step.saved.inputs[0], 2.0 / n);
            const base = try ctx.mul(centered, scale);
            break :blk try ctx.mul(grad_out, base);
        },
        .std_all => blk: {
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const mean = try ctx.meanAll(x);
            const centered = try ctx.sub(x, mean);
            const n = @as(f64, @floatFromInt(step.saved.inputs[0].shape.numel()));
            const stdv = try ctx.sqrt(try ctx.meanAll(try ctx.mul(centered, centered)));
            const scale = try ctx.scalarLike(step.saved.inputs[0], 1.0 / n);
            const numer = try ctx.mul(centered, scale);
            const base = try ctx.div(numer, stdv);
            break :blk try ctx.mul(grad_out, base);
        },
        .sum_axis => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const keepdim = step.keepdim orelse false;
            const expanded = if (keepdim) grad_out else try ctx.unsqueeze(grad_out, axis);
            const ones = try ctx.fullLike(step.saved.inputs[0], 1.0);
            break :blk try ctx.mul(ones, expanded);
        },
        .mean_axis => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const keepdim = step.keepdim orelse false;
            const dim = step.saved.inputs[0].shape.dims[axis];
            if (dim == 0) return error.GradUnsupported;
            const expanded = if (keepdim) grad_out else try ctx.unsqueeze(grad_out, axis);
            const scale = try ctx.fullLike(step.saved.inputs[0], 1.0 / @as(f64, @floatFromInt(dim)));
            break :blk try ctx.mul(scale, expanded);
        },
        .min_axis, .max_axis => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const keepdim = step.keepdim orelse false;
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const expanded_y = if (keepdim) y else try ctx.unsqueeze(y, axis);
            const expanded_grad = if (keepdim) grad_out else try ctx.unsqueeze(grad_out, axis);
            const mask = try ctx.eq(x, expanded_y);
            const zero = try ctx.fullLike(step.saved.inputs[0], 0.0);
            break :blk try ctx.whereSelect(mask, expanded_grad, zero);
        },
        .variance_axis => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const keepdim = step.keepdim orelse false;
            const dim = step.saved.inputs[0].shape.dims[axis];
            if (dim == 0) return error.GradUnsupported;
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const expanded = if (keepdim) grad_out else try ctx.unsqueeze(grad_out, axis);
            const mean = try ctx.meanKeepdim(x, axis);
            const centered = try ctx.sub(x, mean);
            const scale = try ctx.scalarLike(step.saved.inputs[0], 2.0 / @as(f64, @floatFromInt(dim)));
            const base = try ctx.mul(centered, scale);
            break :blk try ctx.mul(expanded, base);
        },
        .std_axis => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const keepdim = step.keepdim orelse false;
            const dim = step.saved.inputs[0].shape.dims[axis];
            if (dim == 0) return error.GradUnsupported;
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const expanded = if (keepdim) grad_out else try ctx.unsqueeze(grad_out, axis);
            const mean = try ctx.meanKeepdim(x, axis);
            const centered = try ctx.sub(x, mean);
            const centered2 = try ctx.mul(centered, centered);
            const stdv = try ctx.sqrt(try ctx.meanKeepdim(centered2, axis));
            const scale = try ctx.scalarLike(step.saved.inputs[0], 1.0 / @as(f64, @floatFromInt(dim)));
            const numer = try ctx.mul(centered, scale);
            const base = try ctx.div(numer, stdv);
            break :blk try ctx.mul(expanded, base);
        },
        .dot => blk: {
            const other = try ctx.bindTensor(@constCast(step.saved.inputs[if (input_slot == 0) 1 else 0]));
            break :blk try ctx.mul(grad_out, other);
        },
        .matmul => blk: {
            if (input_slot == 0) {
                const bt = try transposeLastTwo(ctx, try ctx.bindTensor(@constCast(step.saved.inputs[1])));
                const raw = try ctx.matmul(grad_out, bt);
                break :blk try ctx.reduceToParentShape(raw, step.saved.inputs[0]);
            }
            if (input_slot == 1) {
                const at = try transposeLastTwo(ctx, try ctx.bindTensor(@constCast(step.saved.inputs[0])));
                const raw = try ctx.matmul(at, grad_out);
                break :blk try ctx.reduceToParentShape(raw, step.saved.inputs[1]);
            }
            return error.GradUnsupported;
        },
        .softmax => blk: {
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const gy = try ctx.mul(grad_out, y);
            const axis = step.axis orelse return error.GradUnsupported;
            const dot = try ctx.sumKeepdim(gy, axis);
            const shifted = try ctx.sub(grad_out, dot);
            break :blk try ctx.mul(y, shifted);
        },
        .log_softmax_nll, .cross_entropy => blk: {
            const logits = try ctx.bindTensor(@constCast(step.saved.inputs[0]));
            const targets = try ctx.bindTensor(@constCast(step.saved.inputs[1]));
            const axis = step.axis orelse return error.GradUnsupported;
            const axis_size = step.saved.inputs[0].shape.dims[axis];
            if (axis_size == 0) return error.GradUnsupported;
            const groups = step.saved.inputs[0].shape.numel() / axis_size;
            const scale_value = 1.0 / @as(f64, @floatFromInt(groups));
            if (input_slot == 0) {
                const probs = try ctx.softmax(logits, axis);
                const delta = try ctx.sub(probs, targets);
                const scale = try ctx.fullLike(step.saved.inputs[0], scale_value);
                const scaled = try ctx.mul(delta, scale);
                break :blk try ctx.mul(grad_out, scaled);
            }
            if (input_slot == 1) {
                const log_probs = try ctx.logSoftmax(logits, axis);
                const neg_log_probs = try ctx.neg(log_probs);
                const scale = try ctx.fullLike(step.saved.inputs[1], scale_value);
                const scaled = try ctx.mul(neg_log_probs, scale);
                break :blk try ctx.mul(grad_out, scaled);
            }
            return error.GradUnsupported;
        },
        .cross_entropy_indexed => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const logits_value = step.saved.inputs[0];
            const targets_value = step.saved.inputs[1];
            if (logits_value.shape.rank() != 2) return error.GradUnsupported;
            const rows = logits_value.shape.dims[0];
            if (rows == 0) return error.GradUnsupported;
            const axis = step.axis orelse 1;
            if (axis != 1) return error.GradUnsupported;
            const logits = try ctx.bindTensor(@constCast(logits_value));
            const targets = try ctx.bindTensor(@constCast(targets_value));
            break :blk try ctx.crossEntropyIndexedBackward(logits, targets, grad_out, axis);
        },
        .layer_norm => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const y = try ctx.bindValue(@constCast(step.saved.output orelse return error.GradUnsupported));
            const yy = try ctx.mul(y, y);
            const inv_std = try ctx.sqrt(try ctx.meanKeepdim(yy, axis));
            const dy_mean = try ctx.meanKeepdim(grad_out, axis);
            const dy_y = try ctx.mul(grad_out, y);
            const dy_y_mean = try ctx.meanKeepdim(dy_y, axis);
            const centered = try ctx.sub(grad_out, dy_mean);
            const proj = try ctx.mul(y, dy_y_mean);
            const numer = try ctx.sub(centered, proj);
            break :blk try ctx.mul(inv_std, numer);
        },
        .rms_norm => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const eps = step.scalar_a orelse return error.GradUnsupported;
            const x = try ctx.bindValue(@constCast(step.saved.inputs[0]));
            const xx = try ctx.mul(x, x);
            const mean_x2 = try ctx.meanKeepdim(xx, axis);
            const epsv = try ctx.scalarLike(step.saved.inputs[0], eps);
            const denom = try ctx.sqrt(try ctx.add(mean_x2, epsv));
            const one = try ctx.scalarLike(step.saved.inputs[0], 1.0);
            const inv_rms = try ctx.div(one, denom);
            const main = try ctx.mul(grad_out, inv_rms);
            const dy_x = try ctx.mul(grad_out, x);
            const sum_dy_x = try ctx.sumKeepdim(dy_x, axis);
            const inv_rms2 = try ctx.mul(inv_rms, inv_rms);
            const inv_rms3 = try ctx.mul(inv_rms2, inv_rms);
            const dim = step.saved.inputs[0].shape.dims[axis];
            if (dim == 0) return error.GradUnsupported;
            const inv_n = try ctx.scalarLike(step.saved.inputs[0], 1.0 / @as(f64, @floatFromInt(dim)));
            const coeff = try ctx.mul(inv_rms3, inv_n);
            const corr_scale = try ctx.mul(coeff, sum_dy_x);
            const corr = try ctx.mul(x, corr_scale);
            break :blk try ctx.sub(main, corr);
        },
        .where => blk: {
            const cond = try ctx.bindTensor(@constCast(step.saved.inputs[0]));
            const zero = try ctx.fullLike(step.saved.inputs[input_slot], 0.0);
            if (input_slot == 1) {
                const grad = try ctx.whereSelect(cond, grad_out, zero);
                break :blk try ctx.reduceToParentShape(grad, step.saved.inputs[1]);
            }
            if (input_slot == 2) {
                const grad = try ctx.whereSelect(cond, zero, grad_out);
                break :blk try ctx.reduceToParentShape(grad, step.saved.inputs[2]);
            }
            return error.GradUnsupported;
        },
        .cat => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const rank = ctx.valueSpec(grad_out).shape.dims.len;
            if (axis >= rank) return error.GradUnsupported;
            var start: usize = 0;
            for (0..input_slot) |i| start += step.saved.inputs[i].shape.dims[axis];
            const stop = start + step.saved.inputs[input_slot].shape.dims[axis];
            const ranges = try ctx.getAllocator().alloc(@import("../shared/types/operation/options.zig").SliceRange, rank);
            defer ctx.getAllocator().free(ranges);
            const grad_shape = ctx.valueSpec(grad_out).shape.dims;
            for (0..rank) |d| {
                ranges[d] = if (d == axis)
                    .{ .start = start, .stop = stop, .step = 1 }
                else
                    .{ .start = 0, .stop = grad_shape[d], .step = 1 };
            }
            const sliced = try ctx.slice(grad_out, ranges);
            break :blk try ctx.contiguous(sliced);
        },
        .stack => blk: {
            const axis = step.axis orelse return error.GradUnsupported;
            const rank = ctx.valueSpec(grad_out).shape.dims.len;
            if (axis >= rank) return error.GradUnsupported;
            const ranges = try ctx.getAllocator().alloc(@import("../shared/types/operation/options.zig").SliceRange, rank);
            defer ctx.getAllocator().free(ranges);
            const grad_shape = ctx.valueSpec(grad_out).shape.dims;
            for (0..rank) |d| {
                ranges[d] = if (d == axis)
                    .{ .start = input_slot, .stop = input_slot + 1, .step = 1 }
                else
                    .{ .start = 0, .stop = grad_shape[d], .step = 1 };
            }
            const sliced = try ctx.slice(grad_out, ranges);
            const squeezed = try ctx.squeeze(sliced, axis);
            break :blk try ctx.contiguous(squeezed);
        },
        .slice => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const ranges = step.slice_ranges orelse return error.GradUnsupported;
            const input = step.saved.inputs[0];
            if (input.shape.rank() != 1 or ranges.len != 1) return error.GradUnsupported;
            const range = ranges[0];
            if (range.step <= 0) return error.GradUnsupported;
            const zero = try ctx.fullLike(input, 0.0);
            const updates_len = ctx.valueSpec(grad_out).shape.numel();
            const idx = try ctx.getAllocator().alloc(i64, updates_len);
            defer ctx.getAllocator().free(idx);
            var pos = range.start;
            for (0..updates_len) |i| {
                idx[i] = @intCast(pos);
                pos += @intCast(range.step);
            }
            const device = input.device() orelse return error.InputNotMaterialized;
            const index_id = try ctx.ownedI64(&.{updates_len}, idx, device);
            break :blk try ctx.scatterAdd(zero, index_id, grad_out, 0);
        },
        .gather => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const input = step.saved.inputs[0];
            const index = step.saved.aux orelse step.saved.inputs[1];
            const axis = step.axis orelse return error.GradUnsupported;
            const zero = try ctx.fullLike(input, 0.0);
            const index_id = try ctx.bindTensor(@constCast(index));
            break :blk try ctx.scatterAdd(zero, index_id, grad_out, axis);
        },
        .index_select => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const input = step.saved.inputs[0];
            const index = step.saved.aux orelse step.saved.inputs[1];
            const axis = step.axis orelse return error.GradUnsupported;
            const zero = try ctx.fullLike(input, 0.0);
            const expanded_index = try ctx.expandAxisIndex(index, ctx.valueSpec(grad_out).shape.dims, axis);
            break :blk try ctx.scatterAdd(zero, expanded_index, grad_out, axis);
        },
        .embedding => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const table = step.saved.inputs[0];
            const index = step.saved.aux orelse step.saved.inputs[1];
            if (table.shape.rank() < 2) return error.GradUnsupported;
            const vocab = table.shape.dims[0];
            var emb_dim: usize = 1;
            for (table.shape.dims[1..]) |d| emb_dim *= d;
            const token_count = index.shape.numel();
            if (token_count * emb_dim != ctx.valueSpec(grad_out).shape.numel()) return error.GradUnsupported;
            const zero = try ctx.fullLike(table, 0.0);
            const zero2d = try ctx.reshape(zero, &.{ vocab, emb_dim });
            const updates2d = try ctx.reshape(grad_out, &.{ token_count, emb_dim });
            const expanded_index = try ctx.expandAxisIndex(index, &.{ token_count, emb_dim }, 0);
            const grad2d = try ctx.scatterAdd(zero2d, expanded_index, updates2d, 0);
            break :blk try ctx.reshape(grad2d, table.shape.dims);
        },
        .topk => blk: {
            if (input_slot != 0) return error.GradUnsupported;
            const input = step.saved.inputs[0];
            const indices = step.saved.aux orelse return error.GradUnsupported;
            const axis = step.axis orelse return error.GradUnsupported;
            const zero = try ctx.fullLike(input, 0.0);
            const index_id = try ctx.bindTensor(@constCast(indices));
            break :blk try ctx.scatterAdd(zero, index_id, grad_out, axis);
        },
        else => error.GradUnsupported,
    };
}
