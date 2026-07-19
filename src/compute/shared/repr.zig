const std = @import("std");
const cfg = @import("../../config.zig");
const Tensor = @import("../types/tensor/tensor.zig").Tensor;
const DType = @import("../types/tensor/dtype.zig").DType;

fn reprAlloc() std.mem.Allocator {
    return std.heap.c_allocator;
}

pub fn alloc() std.mem.Allocator {
    return reprAlloc();
}

pub const Mode = enum {
    text,
    html,
};

pub const ReprOpts = struct {
    mode: Mode = .text,
    sparse: bool = false,
    max_rows: usize = 20,
    max_cols: usize = 12,
};

pub fn defaultOpts() ReprOpts {
    return .{
        .max_rows = cfg.config.repr.repr_max_rows,
        .max_cols = cfg.config.repr.repr_max_cols,
    };
}

pub fn compactMaxItems() usize {
    return cfg.config.repr.max_items;
}

pub fn formatCompact(data: *Tensor, is_tensor: bool, requires_grad: bool) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(reprAlloc());

    try buf.appendSlice(reprAlloc(), if (is_tensor) "Tensor(" else "Array(");
    switch (data.dtype) {
        .f32 => try formatCompactByType(f32, &buf, data, compactMaxItems()),
        .f64 => try formatCompactByType(f64, &buf, data, compactMaxItems()),
        .i64 => try formatCompactByType(i64, &buf, data, compactMaxItems()),
    }
    if (is_tensor and requires_grad) try buf.appendSlice(reprAlloc(), ", requires_grad=true");
    try buf.appendSlice(reprAlloc(), ", dtype=");
    try buf.appendSlice(reprAlloc(), dtypeName(data.dtype));
    try buf.append(reprAlloc(), ')');
    return buf.toOwnedSlice(reprAlloc());
}

pub fn formatText(data: *Tensor, is_tensor: bool, opts: ReprOpts) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(reprAlloc());

    if (data.shape.rank() == 0) {
        try appendShapeFooter(&buf, data.shape.dims, data.dtype, is_tensor, data.axes);
        switch (data.dtype) {
            .f64 => {
                const ptr = try readableTypedPtr(f64, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(f64, ptr[0], &tmp));
            },
            .f32 => {
                const ptr = try readableTypedPtr(f32, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(f32, ptr[0], &tmp));
            },
            .i64 => {
                const ptr = try readableTypedPtr(i64, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(i64, ptr[0], &tmp));
            },
        }
        try buf.append(reprAlloc(), '\n');
        return buf.toOwnedSlice(reprAlloc());
    }

    try appendShapeFooter(&buf, data.shape.dims, data.dtype, is_tensor, data.axes);
    switch (data.dtype) {
        .f64 => {
            const ptr = try readableTypedPtr(f64, data);
            if (data.shape.rank() == 1) {
                try formatTerminal1D(f64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try formatTerminalND(f64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            }
        },
        .f32 => {
            const ptr = try readableTypedPtr(f32, data);
            if (data.shape.rank() == 1) {
                try formatTerminal1D(f32, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try formatTerminalND(f32, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            }
        },
        .i64 => {
            const ptr = try readableTypedPtr(i64, data);
            if (data.shape.rank() == 1) {
                try formatTerminal1D(i64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try formatTerminalND(i64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            }
        },
    }
    return buf.toOwnedSlice(reprAlloc());
}

pub fn formatHtml(data: *Tensor, is_tensor: bool, opts: ReprOpts) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(reprAlloc());

    if (data.shape.rank() == 0) {
        try appendHtmlMeta(&buf, data.shape.dims, data.dtype, is_tensor, data.axes);
        try buf.appendSlice(reprAlloc(), "<span class=\"repr-scalar\">");
        switch (data.dtype) {
            .f64 => {
                const ptr = try readableTypedPtr(f64, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(f64, ptr[0], &tmp));
            },
            .f32 => {
                const ptr = try readableTypedPtr(f32, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(f32, ptr[0], &tmp));
            },
            .i64 => {
                const ptr = try readableTypedPtr(i64, data);
                var tmp: [64]u8 = undefined;
                try buf.appendSlice(reprAlloc(), formatValue(i64, ptr[0], &tmp));
            },
        }
        try buf.appendSlice(reprAlloc(), "</span>\n");
        return buf.toOwnedSlice(reprAlloc());
    }

    try appendHtmlMeta(&buf, data.shape.dims, data.dtype, is_tensor, data.axes);
    switch (data.dtype) {
        .f64 => {
            const ptr = try readableTypedPtr(f64, data);
            if (data.shape.rank() >= 2) {
                try formatHtmlND(f64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try buf.appendSlice(reprAlloc(), "<table class=\"repr\">\n");
                try formatHtml1D(f64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
                try buf.appendSlice(reprAlloc(), "</table>\n");
            }
        },
        .f32 => {
            const ptr = try readableTypedPtr(f32, data);
            if (data.shape.rank() >= 2) {
                try formatHtmlND(f32, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try buf.appendSlice(reprAlloc(), "<table class=\"repr\">\n");
                try formatHtml1D(f32, ptr, data.shape.dims, data.layout.strides, opts, &buf);
                try buf.appendSlice(reprAlloc(), "</table>\n");
            }
        },
        .i64 => {
            const ptr = try readableTypedPtr(i64, data);
            if (data.shape.rank() >= 2) {
                try formatHtmlND(i64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
            } else {
                try buf.appendSlice(reprAlloc(), "<table class=\"repr\">\n");
                try formatHtml1D(i64, ptr, data.shape.dims, data.layout.strides, opts, &buf);
                try buf.appendSlice(reprAlloc(), "</table>\n");
            }
        },
    }

    return buf.toOwnedSlice(reprAlloc());
}

fn formatCompactByType(comptime T: type, buf: *std.ArrayList(u8), data: *Tensor, max_items: usize) !void {
    const ptr = try readableTypedPtr(T, data);
    const edge_items = @max(@as(usize, 1), max_items / 2);
    try formatCompactDim(T, buf, ptr, data.shape.dims, data.layout.strides, 0, 0, data.shape.numel() > max_items, edge_items, max_items);
}

fn formatCompactDim(
    comptime T: type,
    buf: *std.ArrayList(u8),
    ptr: [*]const T,
    shape: []const usize,
    strides: []const isize,
    dim: usize,
    offset: usize,
    summarize: bool,
    edge_items: usize,
    max_items: usize,
) !void {
    if (shape.len == 0 or dim >= shape.len) {
        var tmp: [64]u8 = undefined;
        try buf.appendSlice(reprAlloc(), formatValue(T, ptr[offset], &tmp));
        return;
    }

    try buf.append(reprAlloc(), '[');
    const len = shape[dim];
    const last_dim = dim + 1 == shape.len;
    const should_truncate = if (last_dim)
        summarize and len > max_items
    else
        summarize and len > 2;

    if (!should_truncate) {
        for (0..len) |i| {
            if (i > 0) try buf.appendSlice(reprAlloc(), ", ");
            try formatCompactDim(T, buf, ptr, shape, strides, dim + 1, offset + i * strideIndex(strides[dim]), summarize, edge_items, max_items);
        }
    } else if (last_dim) {
        for (0..edge_items) |i| {
            if (i > 0) try buf.appendSlice(reprAlloc(), ", ");
            try formatCompactDim(T, buf, ptr, shape, strides, dim + 1, offset + i * strideIndex(strides[dim]), summarize, edge_items, max_items);
        }
        try buf.appendSlice(reprAlloc(), ", ..., ");
        for (len - edge_items..len) |i| {
            if (i > len - edge_items) try buf.appendSlice(reprAlloc(), ", ");
            try formatCompactDim(T, buf, ptr, shape, strides, dim + 1, offset + i * strideIndex(strides[dim]), summarize, edge_items, max_items);
        }
    } else {
        try formatCompactDim(T, buf, ptr, shape, strides, dim + 1, offset, summarize, edge_items, max_items);
        try buf.appendSlice(reprAlloc(), ", ..., ");
        try formatCompactDim(T, buf, ptr, shape, strides, dim + 1, offset + (len - 1) * strideIndex(strides[dim]), summarize, edge_items, max_items);
    }
    try buf.append(reprAlloc(), ']');
}

fn readableTypedPtr(comptime T: type, data: *Tensor) ![*]const T {
    const readable = try data.storage.?.readableBytes();
    return @as([*]const T, @ptrCast(@alignCast(readable.ptr))) + data.layout.offset;
}

fn formatTerminal1D(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const n = shape[0];
    const truncate = n > opts.max_cols;
    const edge = if (truncate) opts.max_cols / 2 else n;
    const show_count = if (truncate) edge * 2 else n;
    const values = try reprAlloc().alloc([]const u8, show_count);
    defer reprAlloc().free(values);
    const value_bufs = try reprAlloc().alloc([64]u8, show_count);
    defer reprAlloc().free(value_bufs);

    for (0..edge) |i| values[i] = formatValue(T, ptr[i * strideIndex(strides[0])], &value_bufs[i]);
    if (truncate) for (0..edge) |i| {
        const idx = n - edge + i;
        values[edge + i] = formatValue(T, ptr[idx * strideIndex(strides[0])], &value_bufs[edge + i]);
    };

    var max_width: usize = 0;
    for (values) |v| {
        if (v.len > max_width) max_width = v.len;
    }
    var idx_tmp: [16]u8 = undefined;
    const last_idx_str = std.fmt.bufPrint(&idx_tmp, "{d}", .{n - 1}) catch "?";
    const col_width = @max(max_width, last_idx_str.len);

    try buf.appendSlice(reprAlloc(), "  ");
    for (0..edge) |i| {
        var itmp: [16]u8 = undefined;
        try padRight(buf, std.fmt.bufPrint(&itmp, "{d}", .{i}) catch "?", col_width + 2);
    }
    if (truncate) {
        try buf.appendSlice(reprAlloc(), "...  ");
        for (0..edge) |i| {
            var itmp: [16]u8 = undefined;
            const idx = n - edge + i;
            try padRight(buf, std.fmt.bufPrint(&itmp, "{d}", .{idx}) catch "?", col_width + 2);
        }
    }
    try buf.append(reprAlloc(), '\n');

    try buf.appendSlice(reprAlloc(), "[ ");
    for (0..edge) |i| {
        try padLeft(buf, values[i], col_width);
        try buf.appendSlice(reprAlloc(), "  ");
    }
    if (truncate) {
        try buf.appendSlice(reprAlloc(), "...  ");
        for (0..edge) |i| {
            try padLeft(buf, values[edge + i], col_width);
            try buf.appendSlice(reprAlloc(), "  ");
        }
    }
    try buf.appendSlice(reprAlloc(), "]\n");
}

fn formatTerminalND(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const rendered = try renderBlock(T, ptr, shape, strides, opts);
    defer reprAlloc().free(rendered);
    try buf.appendSlice(reprAlloc(), rendered);
}

fn renderBlock(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts) ![]u8 {
    if (shape.len == 2) {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(reprAlloc());
        try formatTerminal2D(T, ptr, shape, strides, opts, &buf);
        return buf.toOwnedSlice(reprAlloc());
    }

    // The current QJS repr parity target only needs 2D sparse terminal output
    // plus 1D terminal formatting; fall back to a readable nested rendering for
    // higher-rank text output until those snapshots are covered.
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(reprAlloc());
    formatNested(T, ptr, shape, strides, 0, 0, &buf);
    try buf.append(reprAlloc(), '\n');
    return buf.toOwnedSlice(reprAlloc());
}

fn formatNested(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, dim: usize, offset: usize, buf: *std.ArrayList(u8)) void {
    if (dim == shape.len) {
        var tmp: [64]u8 = undefined;
        buf.appendSlice(reprAlloc(), formatValue(T, ptr[offset], &tmp)) catch {};
        return;
    }
    buf.append(reprAlloc(), '[') catch {};
    for (0..shape[dim]) |i| {
        if (i > 0) buf.appendSlice(reprAlloc(), ", ") catch {};
        formatNested(T, ptr, shape, strides, dim + 1, offset + i * strideIndex(strides[dim]), buf);
    }
    buf.append(reprAlloc(), ']') catch {};
}

fn formatTerminal2D(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const rows = shape[0];
    const cols = shape[1];
    const trunc_rows = rows > opts.max_rows;
    const trunc_cols = cols > opts.max_cols;
    const row_edge = if (trunc_rows) opts.max_rows / 2 else rows;
    const col_edge = if (trunc_cols) opts.max_cols / 2 else cols;

    var visible_rows: std.ArrayList(usize) = .empty;
    defer visible_rows.deinit(reprAlloc());
    for (0..row_edge) |i| try visible_rows.append(reprAlloc(), i);
    if (trunc_rows) for (rows - row_edge..rows) |i| try visible_rows.append(reprAlloc(), i);

    var visible_cols: std.ArrayList(usize) = .empty;
    defer visible_cols.deinit(reprAlloc());
    for (0..col_edge) |i| try visible_cols.append(reprAlloc(), i);
    if (trunc_cols) for (cols - col_edge..cols) |i| try visible_cols.append(reprAlloc(), i);

    const n_vis_rows = visible_rows.items.len;
    const n_vis_cols = visible_cols.items.len;
    const cell_count = n_vis_rows * n_vis_cols;
    const cell_bufs = try reprAlloc().alloc([64]u8, cell_count);
    defer reprAlloc().free(cell_bufs);
    const cell_strs = try reprAlloc().alloc([]const u8, cell_count);
    defer reprAlloc().free(cell_strs);
    const col_widths = try reprAlloc().alloc(usize, n_vis_cols);
    defer reprAlloc().free(col_widths);

    for (visible_cols.items, 0..) |ci, j| {
        var itmp: [16]u8 = undefined;
        col_widths[j] = (std.fmt.bufPrint(&itmp, "{d}", .{ci}) catch unreachable).len;
    }

    for (visible_rows.items, 0..) |ri, i| {
        for (visible_cols.items, 0..) |ci, j| {
            const val = ptr[ri * strideIndex(strides[0]) + ci * strideIndex(strides[1])];
            const idx = i * n_vis_cols + j;
            cell_strs[idx] = if (opts.sparse and val == 0) "" else formatValue(T, val, &cell_bufs[idx]);
            if (cell_strs[idx].len > col_widths[j]) col_widths[j] = cell_strs[idx].len;
        }
    }

    var ridx_tmp: [16]u8 = undefined;
    const ridx_width = (std.fmt.bufPrint(&ridx_tmp, "{d}", .{if (visible_rows.items.len > 0) visible_rows.items[visible_rows.items.len - 1] else 0}) catch unreachable).len;

    try repeatChar(buf, ' ', ridx_width + 4);
    for (visible_cols.items, 0..) |ci, j| {
        if (j > 0) try buf.appendSlice(reprAlloc(), "  ");
        if (trunc_cols and j == col_edge) try buf.appendSlice(reprAlloc(), "  ⋯   ");
        var itmp: [16]u8 = undefined;
        try padLeft(buf, std.fmt.bufPrint(&itmp, "{d}", .{ci}) catch "?", col_widths[j]);
    }
    try buf.append(reprAlloc(), '\n');

    var inner_content_width: usize = 0;
    for (col_widths, 0..) |w, j| {
        if (j > 0) inner_content_width += 2;
        if (trunc_cols and j == col_edge) inner_content_width += 6;
        inner_content_width += w;
    }

    try repeatChar(buf, ' ', ridx_width + 1);
    try buf.appendSlice(reprAlloc(), "┌");
    try repeatChar(buf, ' ', inner_content_width + 4);
    try buf.appendSlice(reprAlloc(), "┐");
    try buf.append(reprAlloc(), '\n');

    for (visible_rows.items, 0..) |ri, i| {
        if (trunc_rows and i == row_edge) {
            try repeatChar(buf, ' ', ridx_width + 1);
            try buf.appendSlice(reprAlloc(), "│  ");
            for (col_widths, 0..) |w, j| {
                if (j > 0) try buf.appendSlice(reprAlloc(), "  ");
                if (trunc_cols and j == col_edge) try buf.appendSlice(reprAlloc(), "  ⋯   ");
                try padLeft(buf, "⋮", w);
            }
            try buf.appendSlice(reprAlloc(), "  │\n");
        }

        var rtmp: [16]u8 = undefined;
        try padLeft(buf, std.fmt.bufPrint(&rtmp, "{d}", .{ri}) catch "?", ridx_width);
        try buf.appendSlice(reprAlloc(), " │  ");
        for (0..n_vis_cols) |j| {
            if (j > 0) try buf.appendSlice(reprAlloc(), "  ");
            if (trunc_cols and j == col_edge) try buf.appendSlice(reprAlloc(), "  ⋯   ");
            try padLeft(buf, cell_strs[i * n_vis_cols + j], col_widths[j]);
        }
        try buf.appendSlice(reprAlloc(), "  │\n");
    }

    try repeatChar(buf, ' ', ridx_width + 1);
    try buf.appendSlice(reprAlloc(), "└");
    try repeatChar(buf, ' ', inner_content_width + 4);
    try buf.appendSlice(reprAlloc(), "┘");
    try buf.append(reprAlloc(), '\n');
}

fn formatHtml1D(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const n = shape[0];
    const trunc = n > opts.max_cols;
    const edge = if (trunc) opts.max_cols / 2 else n;

    try buf.appendSlice(reprAlloc(), "<tr><th></th>");
    for (0..edge) |i| {
        try buf.print(reprAlloc(), "<th>{d}</th>", .{i});
    }
    if (trunc) try buf.appendSlice(reprAlloc(), "<th>⋯</th>");
    if (trunc) {
        for (n - edge..n) |i| try buf.print(reprAlloc(), "<th>{d}</th>", .{i});
    }
    try buf.appendSlice(reprAlloc(), "</tr>\n<tr><td></td>");
    for (0..edge) |i| {
        var tmp: [64]u8 = undefined;
        try buf.print(reprAlloc(), "<td>{s}</td>", .{formatValue(T, ptr[i * strideIndex(strides[0])], &tmp)});
    }
    if (trunc) try buf.appendSlice(reprAlloc(), "<td>⋯</td>");
    if (trunc) {
        for (n - edge..n) |i| {
            var tmp: [64]u8 = undefined;
            try buf.print(reprAlloc(), "<td>{s}</td>", .{formatValue(T, ptr[i * strideIndex(strides[0])], &tmp)});
        }
    }
    try buf.appendSlice(reprAlloc(), "</tr>\n");
}

fn formatHtmlND(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const ndim = shape.len;
    if (ndim == 2) {
        try buf.appendSlice(reprAlloc(), "<table class=\"repr\">\n");
        try formatHtml2D(T, ptr, shape, strides, opts, buf);
        try buf.appendSlice(reprAlloc(), "</table>\n");
        return;
    }

    try buf.appendSlice(reprAlloc(), "<table class=\"repr repr-nd\">\n");
    const n_outer_dims = ndim - 2;
    const inner_shape = shape[ndim - 2 .. ndim];
    const inner_strides = strides[ndim - 2 .. ndim];
    var n_slices: usize = 1;
    for (0..n_outer_dims) |d| n_slices *= shape[d];
    const trunc = n_slices > opts.max_rows;
    const edge = if (trunc) opts.max_rows / 2 else n_slices;
    var multi_idx: [16]usize = undefined;

    for (0..n_slices) |linear_idx| {
        if (trunc and linear_idx >= edge and linear_idx < n_slices - edge) {
            if (linear_idx == edge) try buf.appendSlice(reprAlloc(), "<tr><td class=\"repr-idx\">⋮</td><td>⋮</td></tr>\n");
            continue;
        }

        var remaining = linear_idx;
        var dim_i: usize = n_outer_dims;
        while (dim_i > 0) {
            dim_i -= 1;
            multi_idx[dim_i] = remaining % shape[dim_i];
            remaining /= shape[dim_i];
        }

        var base_offset: usize = 0;
        for (0..n_outer_dims) |d| base_offset += multi_idx[d] * strideIndex(strides[d]);

        try buf.appendSlice(reprAlloc(), "<tr><td class=\"repr-idx\">");
        if (n_outer_dims == 1) {
            try buf.print(reprAlloc(), "{d}", .{multi_idx[0]});
        } else {
            try buf.appendSlice(reprAlloc(), "(");
            for (0..n_outer_dims) |d| {
                if (d > 0) try buf.appendSlice(reprAlloc(), ",");
                try buf.print(reprAlloc(), "{d}", .{multi_idx[d]});
            }
            try buf.appendSlice(reprAlloc(), ")");
        }
        try buf.appendSlice(reprAlloc(), "</td><td>\n<table class=\"repr repr-inner\">\n");
        try formatHtml2D(T, ptr + base_offset, inner_shape, inner_strides, opts, buf);
        try buf.appendSlice(reprAlloc(), "</table>\n</td></tr>\n");
    }
    try buf.appendSlice(reprAlloc(), "</table>\n");
}

fn formatHtml2D(comptime T: type, ptr: [*]const T, shape: []const usize, strides: []const isize, opts: ReprOpts, buf: *std.ArrayList(u8)) !void {
    const rows = shape[0];
    const cols = shape[1];
    const trunc_rows = rows > opts.max_rows;
    const trunc_cols = cols > opts.max_cols;
    const row_edge = if (trunc_rows) opts.max_rows / 2 else rows;
    const col_edge = if (trunc_cols) opts.max_cols / 2 else cols;

    try buf.appendSlice(reprAlloc(), "<tr><th></th>");
    for (0..col_edge) |j| try buf.print(reprAlloc(), "<th>{d}</th>", .{j});
    if (trunc_cols) try buf.appendSlice(reprAlloc(), "<th>⋯</th>");
    if (trunc_cols) for (cols - col_edge..cols) |j| try buf.print(reprAlloc(), "<th>{d}</th>", .{j});
    try buf.appendSlice(reprAlloc(), "</tr>\n");

    for (0..rows) |ri| {
        if (trunc_rows and ri >= row_edge and ri < rows - row_edge) {
            if (ri == row_edge) {
                try buf.appendSlice(reprAlloc(), "<tr><td class=\"repr-idx\">⋮</td>");
                for (0..col_edge) |_| try buf.appendSlice(reprAlloc(), "<td>⋮</td>");
                if (trunc_cols) try buf.appendSlice(reprAlloc(), "<td>⋮</td>");
                if (trunc_cols) for (0..col_edge) |_| try buf.appendSlice(reprAlloc(), "<td>⋮</td>");
                try buf.appendSlice(reprAlloc(), "</tr>\n");
            }
            continue;
        }

        try buf.print(reprAlloc(), "<tr><td class=\"repr-idx\">{d}</td>", .{ri});
        for (0..col_edge) |j| try appendHtmlCell(T, buf, ptr[ri * strideIndex(strides[0]) + j * strideIndex(strides[1])], opts.sparse);
        if (trunc_cols) try buf.appendSlice(reprAlloc(), "<td>⋯</td>");
        if (trunc_cols) for (cols - col_edge..cols) |j| try appendHtmlCell(T, buf, ptr[ri * strideIndex(strides[0]) + j * strideIndex(strides[1])], opts.sparse);
        try buf.appendSlice(reprAlloc(), "</tr>\n");
    }
}

fn appendHtmlCell(comptime T: type, buf: *std.ArrayList(u8), value: T, sparse: bool) !void {
    if (sparse and value == 0) {
        try buf.appendSlice(reprAlloc(), "<td></td>");
        return;
    }
    var tmp: [64]u8 = undefined;
    try buf.print(reprAlloc(), "<td>{s}</td>", .{formatValue(T, value, &tmp)});
}

fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .f32 => "f32",
        .f64 => "f64",
        .i64 => "i64",
    };
}

fn formatValue(comptime T: type, val: T, buf: *[64]u8) []const u8 {
    if (T == i64) return std.fmt.bufPrint(buf, "{d}", .{val}) catch "?";
    if (@floor(val) == val and @abs(val) < 1e15) {
        const int_val: i64 = @intFromFloat(val);
        return std.fmt.bufPrint(buf, "{d}", .{int_val}) catch "?";
    }
    const s = std.fmt.bufPrint(buf, "{d:.6}", .{val}) catch "?";
    var end = s.len;
    if (std.mem.indexOf(u8, s, ".")) |_| {
        while (end > 0 and s[end - 1] == '0') end -= 1;
        if (end > 0 and s[end - 1] == '.') end += 1;
    }
    return s[0..end];
}

fn appendShapeFooter(buf: *std.ArrayList(u8), shape: []const usize, dtype: DType, is_tensor: bool, axes: ?[]const []const u8) !void {
    try buf.append(reprAlloc(), '[');
    if (shape.len == 0) {
        try buf.appendSlice(reprAlloc(), "Scalar");
    } else {
        for (shape, 0..) |s, i| {
            if (i > 0) try buf.appendSlice(reprAlloc(), "×");
            try buf.print(reprAlloc(), "{d}", .{s});
        }
        try buf.appendSlice(reprAlloc(), if (is_tensor) " Tensor" else " Array");
    }
    try buf.appendSlice(reprAlloc(), ", dtype=");
    try buf.appendSlice(reprAlloc(), dtypeName(dtype));
    try appendTextAxes(buf, axes);
    try buf.appendSlice(reprAlloc(), "]\n");
}

fn appendTextAxes(buf: *std.ArrayList(u8), axes: ?[]const []const u8) !void {
    const names = axes orelse return;
    if (names.len == 0) return;
    try buf.appendSlice(reprAlloc(), ", axes=[");
    for (names, 0..) |axis_name, i| {
        if (i > 0) try buf.appendSlice(reprAlloc(), ", ");
        try buf.appendSlice(reprAlloc(), axis_name);
    }
    try buf.append(reprAlloc(), ']');
}

fn appendHtmlMeta(buf: *std.ArrayList(u8), shape: []const usize, dtype: DType, is_tensor: bool, axes: ?[]const []const u8) !void {
    const names = axes orelse return;
    if (names.len == 0) return;

    try buf.appendSlice(reprAlloc(), "<div class=\"repr-meta\">");
    if (shape.len == 0) {
        try buf.appendSlice(reprAlloc(), "Scalar");
    } else {
        for (shape, 0..) |s, i| {
            if (i > 0) try buf.appendSlice(reprAlloc(), "×");
            try buf.print(reprAlloc(), "{d}", .{s});
        }
        try buf.appendSlice(reprAlloc(), if (is_tensor) " Tensor" else " Array");
    }
    try buf.appendSlice(reprAlloc(), ", dtype=");
    try buf.appendSlice(reprAlloc(), dtypeName(dtype));
    try buf.appendSlice(reprAlloc(), ", axes=[");
    for (names, 0..) |axis_name, i| {
        if (i > 0) try buf.appendSlice(reprAlloc(), ", ");
        try appendHtmlEscaped(buf, axis_name);
    }
    try buf.appendSlice(reprAlloc(), "]</div>\n");
}

fn appendHtmlEscaped(buf: *std.ArrayList(u8), text: []const u8) !void {
    for (text) |byte| {
        switch (byte) {
            '&' => try buf.appendSlice(reprAlloc(), "&amp;"),
            '<' => try buf.appendSlice(reprAlloc(), "&lt;"),
            '>' => try buf.appendSlice(reprAlloc(), "&gt;"),
            '"' => try buf.appendSlice(reprAlloc(), "&quot;"),
            '\'' => try buf.appendSlice(reprAlloc(), "&#39;"),
            else => try buf.append(reprAlloc(), byte),
        }
    }
}

fn padLeft(buf: *std.ArrayList(u8), s: []const u8, width: usize) !void {
    const vis_w = visibleWidth(s);
    if (vis_w < width) try repeatChar(buf, ' ', width - vis_w);
    try buf.appendSlice(reprAlloc(), s);
}

fn padRight(buf: *std.ArrayList(u8), s: []const u8, width: usize) !void {
    try buf.appendSlice(reprAlloc(), s);
    const vis_w = visibleWidth(s);
    if (vis_w < width) try repeatChar(buf, ' ', width - vis_w);
}

fn repeatChar(buf: *std.ArrayList(u8), char: u8, count: usize) !void {
    for (0..count) |_| try buf.append(reprAlloc(), char);
}

fn visibleWidth(line: []const u8) usize {
    var w: usize = 0;
    var i: usize = 0;
    while (i < line.len) {
        const byte = line[i];
        if (byte < 0x80) {
            w += 1;
            i += 1;
        } else if (byte < 0xC0) {
            i += 1;
        } else if (byte < 0xE0) {
            w += 1;
            i += 2;
        } else if (byte < 0xF0) {
            w += 1;
            i += 3;
        } else {
            w += 1;
            i += 4;
        }
    }
    return w;
}

fn strideIndex(stride: isize) usize {
    std.debug.assert(stride >= 0);
    return @intCast(stride);
}
