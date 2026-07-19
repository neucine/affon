/// Data pipeline: declarative step execution on columnar DataFrames.
///
/// Pipeline phases:
///
///   read('data.csv')                  ─┐
///     .drop('id')                      │ BUILD: accumulate steps,
///     .encode('category', 'label')     │        share DataFrame (refcounted),
///     .normalize('x1', 'x2')           │        no data cloned or transformed.
///     .features('x1', 'x2')            │
///     .target('category')             ─┘
///                                      │
///                               ┌──────┴──────┐
///                               │  OPTIMIZE   │ (TODO) column pruning,
///                               │  rewrite    │ step reordering,
///                               │  step list  │ transform fusion.
///                               └──────┬──────┘
///                                      │
///     .toTensors()                    ─┤ EXECUTE: clone once, apply steps,
///                                      │         build tensors.
///                                      ▼
///                                  { X, y }
///
/// Data flow:
///
///   CSV File
///       │
///   ┌───┴────────┐
///   │  csv.parse  │
///   └───┬────────┘
///       │
///   all numeric?──no──→ parseMixed
///       │                  │
///       ▼                  ▼
///   parseNumeric       ┌──────────┐
///       │              │ columns: │ numeric []f64
///       ▼              │          │ categorical {categories, indices}
///   ┌──────────┐       │ row_major: null
///   │ columns: │       └────┬─────┘
///   │  []f64   │            │
///   │ row_major│            │
///   │  []f64   │            │
///   └────┬─────┘            │
///        └──────┬───────────┘
///               ▼
///           SharedDf ←── refcount shared across Dataset instances
///               │
///     ┌─────────┼─────────────┐
///     │         │             │
///  Path A    Path B        Path C
///  no steps  has steps     has steps
///  row_major               + features/target
///     │         │             │
///     ▼         ▼             ▼
///  memcpy    clone df      clone df
///  from      apply steps   apply steps
///  row_major in-place:     in-place
///     │      select/drop      │
///     │      encode        2× toFlatBuffer
///     │      normalize     (X cols, y col)
///     │      standardize      │
///     │      log/clip/...     │
///     │         │             │
///     │      toFlatBuffer     │
///     │      row-by-row       │
///     │      col→row major    │
///     │         │             │
///     └────┬────┘─────────────┘
///          ▼
///   compute tensor wrapper from owned storage (zero-copy)
///          │
///          ▼
///     JSC Tensor
///
const std = @import("std");
const compat = @import("../../support/compat.zig");
const DataFrame = @import("DataFrame.zig").DataFrame;

const Allocator = std.mem.Allocator;

pub const Step = union(enum) {
    select: [][]const u8,
    drop: [][]const u8,
    encode_label: []const u8,
    encode_onehot: []const u8,
    normalize: [][]const u8,
    standardize: [][]const u8,
    log_transform: [][]const u8,
    clip: ClipOp,
    fillna: FillNaOp,
    dropna: [][]const u8,
    rename: RenameOp,
    shuffle: void,
    sample: usize,
    features: [][]const u8,
    target: []const u8,
};

pub const ClipOp = struct {
    columns: [][]const u8,
    min_val: f64,
    max_val: f64,
};

pub const FillNaOp = struct {
    column: []const u8,
    value: f64,
};

pub const RenameOp = struct {
    old_name: []const u8,
    new_name: []const u8,
};

pub const PipelineError = error{
    ColumnNotFound,
    NotNumericColumn,
    NotCategoricalColumn,
    OutOfMemory,
};

/// Execute all steps on a DataFrame, returning a new DataFrame.
/// The input DataFrame is NOT modified.
pub fn execute(_: Allocator, source: *const DataFrame, steps: []const Step) PipelineError!*DataFrame {
    var df = source.clone() catch return PipelineError.OutOfMemory;
    errdefer df.deinit();

    for (steps) |step| {
        switch (step) {
            .select => |cols| {
                try execSelect(df, cols);
                invalidateRowMajor(df);
            },
            .drop => |cols| {
                try execDrop(df, cols);
                invalidateRowMajor(df);
            },
            .encode_label => |col| {
                try execEncodeLabel(df, col);
                invalidateRowMajor(df);
            },
            .encode_onehot => |col| {
                try execEncodeOnehot(df, col);
                invalidateRowMajor(df);
            },
            .normalize => |cols| {
                try execNormalize(df, cols);
                invalidateRowMajor(df);
            },
            .standardize => |cols| {
                try execStandardize(df, cols);
                invalidateRowMajor(df);
            },
            .log_transform => |cols| {
                try execLog(df, cols);
                invalidateRowMajor(df);
            },
            .clip => |op| {
                try execClip(df, op);
                invalidateRowMajor(df);
            },
            .fillna => |op| {
                try execFillNa(df, op);
                invalidateRowMajor(df);
            },
            .dropna => |cols| {
                try execDropNa(df, cols);
                invalidateRowMajor(df);
            },
            .rename => |op| {
                try execRename(df, op);
            },
            .shuffle => {
                execShuffle(df) catch return PipelineError.OutOfMemory;
                invalidateRowMajor(df);
            },
            .sample => |n| {
                execSample(df, n);
                invalidateRowMajor(df);
            },
            .features, .target => {},
        }
    }

    return df;
}

fn invalidateRowMajor(df: *DataFrame) void {
    if (df.row_major) |rm| {
        DataFrame.freeRowMajor(df.allocator, rm);
        df.row_major = null;
    }
}

// --- Step implementations ---

fn execSelect(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    const allocator = df.allocator;

    // Find indices of selected columns
    const keep = allocator.alloc(usize, col_names.len) catch return PipelineError.OutOfMemory;
    defer allocator.free(keep);
    for (col_names, 0..) |name, i| {
        keep[i] = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
    }

    // Build new columns and names
    const new_cols = DataFrame.allocColumns(allocator, col_names.len) catch return PipelineError.OutOfMemory;
    const new_names = DataFrame.allocColumnNames(allocator, col_names.len) catch return PipelineError.OutOfMemory;

    for (keep, 0..) |src_idx, dst_idx| {
        new_cols[dst_idx] = df.columns[src_idx];
        new_names[dst_idx] = df.col_names[src_idx];
    }

    // Free dropped columns and names
    for (df.columns, df.col_names, 0..) |col, name, i| {
        var is_kept = false;
        for (keep) |k| {
            if (k == i) {
                is_kept = true;
                break;
            }
        }
        if (!is_kept) {
            freeColumn(allocator, col);
            DataFrame.freeColumnName(allocator, name);
        }
    }

    DataFrame.freeColumns(allocator, df.columns);
    DataFrame.freeColumnNames(allocator, df.col_names);
    df.columns = new_cols;
    df.col_names = new_names;
}

fn execDrop(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    const allocator = df.allocator;

    // Find indices to drop
    var drop_set = std.StaticBitSet(256).initEmpty();
    for (col_names) |name| {
        const idx = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
        drop_set.set(idx);
    }

    const new_count = df.numCols() - col_names.len;
    const new_cols = DataFrame.allocColumns(allocator, new_count) catch return PipelineError.OutOfMemory;
    const new_names = DataFrame.allocColumnNames(allocator, new_count) catch return PipelineError.OutOfMemory;

    var dst: usize = 0;
    for (df.columns, df.col_names, 0..) |col, name, i| {
        if (drop_set.isSet(i)) {
            freeColumn(allocator, col);
            DataFrame.freeColumnName(allocator, name);
        } else {
            new_cols[dst] = col;
            new_names[dst] = name;
            dst += 1;
        }
    }

    DataFrame.freeColumns(allocator, df.columns);
    DataFrame.freeColumnNames(allocator, df.col_names);
    df.columns = new_cols;
    df.col_names = new_names;
}

fn execEncodeLabel(df: *DataFrame, col_name: []const u8) PipelineError!void {
    const idx = df.findColumn(col_name) orelse return PipelineError.ColumnNotFound;
    switch (df.columns[idx]) {
        .categorical => |cat| {
            // Convert indices to f64 numeric column
            const data = DataFrame.allocNumericColumn(df.allocator, df.num_rows) catch return PipelineError.OutOfMemory;
            for (cat.indices, 0..) |v, i| {
                data[i] = @floatFromInt(v);
            }
            // Free categorical data
            for (cat.categories) |s| DataFrame.freeCategoryName(df.allocator, s);
            DataFrame.freeCategoryNames(df.allocator, cat.categories);
            DataFrame.freeCategoryIndices(df.allocator, cat.indices);
            df.columns[idx] = .{ .numeric = data };
        },
        .numeric => {}, // already numeric, no-op
    }
}

fn execEncodeOnehot(df: *DataFrame, col_name: []const u8) PipelineError!void {
    const allocator = df.allocator;
    const col_idx = df.findColumn(col_name) orelse return PipelineError.ColumnNotFound;

    const cat = switch (df.columns[col_idx]) {
        .categorical => |cat| cat,
        .numeric => return PipelineError.NotCategoricalColumn,
    };

    const num_cats = cat.categories.len;
    const num_rows = df.num_rows;

    // Create N new numeric columns (one per category)
    const new_col_count = df.numCols() - 1 + num_cats;
    const new_cols = DataFrame.allocColumns(allocator, new_col_count) catch return PipelineError.OutOfMemory;
    const new_names = DataFrame.allocColumnNames(allocator, new_col_count) catch return PipelineError.OutOfMemory;

    // Copy columns before the one-hot column
    var dst: usize = 0;
    for (0..col_idx) |i| {
        new_cols[dst] = df.columns[i];
        new_names[dst] = df.col_names[i];
        dst += 1;
    }

    // Generate one-hot columns
    for (0..num_cats) |c| {
        const data = DataFrame.allocNumericColumn(allocator, num_rows) catch return PipelineError.OutOfMemory;
        for (cat.indices, 0..) |row_cat, r| {
            data[r] = if (row_cat == c) 1.0 else 0.0;
        }
        new_cols[dst] = .{ .numeric = data };
        // Name: "species_setosa", "species_versicolor", etc.
        const base_name = df.col_names[col_idx];
        const cat_name = cat.categories[c];
        const name = DataFrame.allocColumnNameFormat(allocator, "{s}_{s}", .{ base_name, cat_name }) catch return PipelineError.OutOfMemory;
        new_names[dst] = name;
        dst += 1;
    }

    // Copy columns after the one-hot column
    for (col_idx + 1..df.numCols()) |i| {
        new_cols[dst] = df.columns[i];
        new_names[dst] = df.col_names[i];
        dst += 1;
    }

    // Free the original categorical column and its name
    for (cat.categories) |s| DataFrame.freeCategoryName(allocator, s);
    DataFrame.freeCategoryNames(allocator, cat.categories);
    DataFrame.freeCategoryIndices(allocator, cat.indices);
    DataFrame.freeColumnName(allocator, df.col_names[col_idx]);

    DataFrame.freeColumns(allocator, df.columns);
    DataFrame.freeColumnNames(allocator, df.col_names);
    df.columns = new_cols;
    df.col_names = new_names;
}

fn execNormalize(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    for (col_names) |name| {
        const idx = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
        const data = switch (df.columns[idx]) {
            .numeric => |d| d,
            .categorical => return PipelineError.NotNumericColumn,
        };

        // Find min and max
        var min_val: f64 = std.math.inf(f64);
        var max_val: f64 = -std.math.inf(f64);
        for (0..df.num_rows) |i| {
            const ri = df.rowIndex(i);
            const v = data[ri];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }

        const range = max_val - min_val;
        if (range == 0.0) continue;

        // Normalize in-place
        for (data) |*v| {
            v.* = (v.* - min_val) / range;
        }
    }
}

fn execStandardize(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    for (col_names) |name| {
        const idx = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
        const data = switch (df.columns[idx]) {
            .numeric => |d| d,
            .categorical => return PipelineError.NotNumericColumn,
        };

        const n: f64 = @floatFromInt(df.num_rows);

        // Compute mean
        var sum: f64 = 0.0;
        for (0..df.num_rows) |i| {
            sum += data[df.rowIndex(i)];
        }
        const mean = sum / n;

        // Compute std
        var sq_sum: f64 = 0.0;
        for (0..df.num_rows) |i| {
            const d = data[df.rowIndex(i)] - mean;
            sq_sum += d * d;
        }
        const std_val = @sqrt(sq_sum / n);
        if (std_val == 0.0) continue;

        // Standardize in-place
        for (data) |*v| {
            v.* = (v.* - mean) / std_val;
        }
    }
}

fn execLog(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    for (col_names) |name| {
        const idx = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
        const data = switch (df.columns[idx]) {
            .numeric => |d| d,
            .categorical => return PipelineError.NotNumericColumn,
        };

        for (data) |*v| {
            v.* = @log(v.*);
        }
    }
}

fn execClip(df: *DataFrame, op: ClipOp) PipelineError!void {
    for (op.columns) |name| {
        const idx = df.findColumn(name) orelse return PipelineError.ColumnNotFound;
        const data = switch (df.columns[idx]) {
            .numeric => |d| d,
            .categorical => return PipelineError.NotNumericColumn,
        };

        for (data) |*v| {
            v.* = @max(op.min_val, @min(op.max_val, v.*));
        }
    }
}

fn execShuffle(df: *DataFrame) !void {
    try df.ensureIndex();
    const idx = df.index.?;
    // Fisher-Yates
    var rng = std.Random.DefaultPrng.init(@truncate(@as(u128, @bitCast(compat.nanoTimestamp()))));
    const random = rng.random();
    var i: usize = df.num_rows;
    while (i > 1) {
        i -= 1;
        const j = random.uintLessThan(usize, i + 1);
        const tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

fn execSample(df: *DataFrame, n: usize) void {
    df.num_rows = @min(n, df.num_rows);
}

fn execFillNa(df: *DataFrame, op: FillNaOp) PipelineError!void {
    const idx = df.findColumn(op.column) orelse return PipelineError.ColumnNotFound;
    const data = switch (df.columns[idx]) {
        .numeric => |d| d,
        .categorical => return PipelineError.NotNumericColumn,
    };
    for (0..df.num_rows) |i| {
        const ri = df.rowIndex(i);
        if (std.math.isNan(data[ri])) data[ri] = op.value;
    }
}

fn execDropNa(df: *DataFrame, col_names: []const []const u8) PipelineError!void {
    // Build set of column indices to check
    const check_cols = if (col_names.len > 0) col_names else df.col_names;

    var col_indices: [256]usize = undefined;
    const n_check = @min(check_cols.len, 256);
    for (0..n_check) |i| {
        col_indices[i] = df.findColumn(check_cols[i]) orelse return PipelineError.ColumnNotFound;
    }

    // Use index to mark which rows to keep
    df.ensureIndex() catch return PipelineError.OutOfMemory;
    const idx = df.index.?;

    var write: usize = 0;
    for (0..df.num_rows) |r| {
        const ri = idx[r];
        var has_nan = false;
        for (col_indices[0..n_check]) |ci| {
            switch (df.columns[ci]) {
                .numeric => |data| {
                    if (std.math.isNan(data[ri])) {
                        has_nan = true;
                        break;
                    }
                },
                .categorical => {},
            }
        }
        if (!has_nan) {
            idx[write] = ri;
            write += 1;
        }
    }
    df.num_rows = write;
}

fn execRename(df: *DataFrame, op: RenameOp) PipelineError!void {
    const idx = df.findColumn(op.old_name) orelse return PipelineError.ColumnNotFound;
    const allocator = df.allocator;
    DataFrame.freeColumnName(allocator, df.col_names[idx]);
    df.col_names[idx] = DataFrame.dupeColumnName(allocator, op.new_name) catch return PipelineError.OutOfMemory;
}

// --- Helpers ---

fn freeColumn(allocator: Allocator, col: DataFrame.Column) void {
    switch (col) {
        .numeric => |data| DataFrame.freeNumericColumn(allocator, data),
        .categorical => |cat| {
            for (cat.categories) |s| DataFrame.freeCategoryName(allocator, s);
            DataFrame.freeCategoryNames(allocator, cat.categories);
            DataFrame.freeCategoryIndices(allocator, cat.indices);
        },
    }
}

// --- Terminal operations ---

/// Build a flat row-major f64 buffer from selected columns, respecting index permutation.
/// Returns owned slice and the width (number of f64 values per row).
pub fn toFlatBuffer(allocator: Allocator, df: *const DataFrame, col_indices: []const usize) !struct { data: []f64, width: usize } {
    // Fast path: if row_major data exists, no index permutation, and all columns selected in order
    if (df.row_major) |rm| {
        if (df.index == null and col_indices.len == df.numCols()) {
            var sequential = true;
            for (col_indices, 0..) |ci, i| {
                if (ci != i) {
                    sequential = false;
                    break;
                }
            }
            if (sequential) {
                // Zero-copy: transfer ownership of row_major slice
                const data = try allocator.dupe(f64, rm[0 .. df.num_rows * df.numCols()]);
                return .{ .data = data, .width = df.numCols() };
            }
        }
    }

    // Calculate width (onehot columns expand)
    var width: usize = 0;
    for (col_indices) |ci| {
        switch (df.columns[ci]) {
            .numeric => width += 1,
            .categorical => |cat| width += cat.categories.len,
        }
    }

    const total = df.num_rows * width;
    const data = try allocator.alloc(f64, total);

    for (0..df.num_rows) |row| {
        const ri = df.rowIndex(row);
        var offset = row * width;
        for (col_indices) |ci| {
            switch (df.columns[ci]) {
                .numeric => |col_data| {
                    data[offset] = col_data[ri];
                    offset += 1;
                },
                .categorical => |cat| {
                    const cat_idx = cat.indices[ri];
                    for (0..cat.categories.len) |c| {
                        data[offset] = if (cat_idx == c) 1.0 else 0.0;
                        offset += 1;
                    }
                },
            }
        }
    }

    return .{ .data = data, .width = width };
}

/// Build schema: column name → tensor column indices
pub fn buildSchema(allocator: Allocator, df: *const DataFrame, col_indices: []const usize) !SchemaResult {
    var entries = std.ArrayList(SchemaEntry).empty;
    defer entries.deinit(allocator);

    var tensor_idx: usize = 0;
    for (col_indices) |ci| {
        const name = df.col_names[ci];
        switch (df.columns[ci]) {
            .numeric => {
                const indices = try allocator.alloc(usize, 1);
                indices[0] = tensor_idx;
                try entries.append(allocator, .{ .name = try allocator.dupe(u8, name), .indices = indices });
                tensor_idx += 1;
            },
            .categorical => |cat| {
                const indices = try allocator.alloc(usize, cat.categories.len);
                for (0..cat.categories.len) |j| {
                    indices[j] = tensor_idx;
                    tensor_idx += 1;
                }
                try entries.append(allocator, .{ .name = try allocator.dupe(u8, name), .indices = indices });
            },
        }
    }

    return .{
        .entries = try entries.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

pub const SchemaEntry = struct {
    name: []const u8,
    indices: []usize,
};

pub const SchemaResult = struct {
    entries: []SchemaEntry,
    allocator: Allocator,

    pub fn deinit(self: *SchemaResult) void {
        for (self.entries) |e| {
            self.allocator.free(e.name);
            self.allocator.free(e.indices);
        }
        self.allocator.free(self.entries);
    }
};

/// Concatenate two DataFrames vertically (same columns required).
/// Returns a new DataFrame owning all data.
pub fn concatDataFrames(allocator: Allocator, a: *const DataFrame, b: *const DataFrame) !*DataFrame {
    const num_cols = a.numCols();
    if (num_cols != b.numCols()) return error.OutOfMemory; // column count mismatch
    const total_rows = a.num_rows + b.num_rows;

    const df = try DataFrame.init(allocator, total_rows, num_cols);
    errdefer df.deinit();

    // Clone column names from `a` and verify `b` matches
    for (a.col_names, b.col_names, 0..) |name_a, name_b, i| {
        if (!std.mem.eql(u8, name_a, name_b)) {
            // Column name mismatch — clean up already-duped names
            for (0..i) |j| DataFrame.freeColumnName(allocator, df.col_names[j]);
            df.deinitShallow();
            return error.OutOfMemory; // mismatched column names
        }
        df.col_names[i] = try DataFrame.dupeColumnName(allocator, name_a);
    }

    // Concatenate column data
    for (a.columns, b.columns, 0..) |col_a, col_b, ci| {
        switch (col_a) {
            .numeric => |data_a| {
                const data_b = switch (col_b) {
                    .numeric => |d| d,
                    .categorical => {
                        df.deinit();
                        return error.OutOfMemory; // type mismatch
                    },
                };
                const data = try DataFrame.allocNumericColumn(allocator, total_rows);
                for (0..a.num_rows) |r| data[r] = data_a[a.rowIndex(r)];
                for (0..b.num_rows) |r| data[a.num_rows + r] = data_b[b.rowIndex(r)];
                df.columns[ci] = .{ .numeric = data };
            },
            .categorical => |cat_a| {
                const cat_b = switch (col_b) {
                    .categorical => |c2| c2,
                    .numeric => {
                        df.deinit();
                        return error.OutOfMemory; // type mismatch
                    },
                };

                // Merge category tables: start with a's categories, add new ones from b
                var cat_list = std.ArrayList([]const u8).empty;
                defer cat_list.deinit(allocator);

                for (cat_a.categories) |s| {
                    try cat_list.append(allocator, try DataFrame.dupeCategoryName(allocator, s));
                }

                // Map b's category indices to merged indices
                const b_remap = try DataFrame.allocCategoryIndices(allocator, cat_b.categories.len);
                defer DataFrame.freeCategoryIndices(allocator, b_remap);
                for (cat_b.categories, 0..) |s, j| {
                    // Linear scan to find existing category
                    var found: ?u32 = null;
                    for (cat_list.items, 0..) |existing, k| {
                        if (std.mem.eql(u8, existing, s)) {
                            found = @intCast(k);
                            break;
                        }
                    }
                    if (found) |idx| {
                        b_remap[j] = idx;
                    } else {
                        b_remap[j] = @intCast(cat_list.items.len);
                        try cat_list.append(allocator, try DataFrame.dupeCategoryName(allocator, s));
                    }
                }

                // Build merged indices
                const indices = try DataFrame.allocCategoryIndices(allocator, total_rows);
                for (0..a.num_rows) |r| {
                    indices[r] = cat_a.indices[a.rowIndex(r)];
                }
                for (0..b.num_rows) |r| {
                    indices[a.num_rows + r] = b_remap[cat_b.indices[b.rowIndex(r)]];
                }

                const category_names = try DataFrame.allocCategoryNames(allocator, cat_list.items.len);
                @memcpy(category_names, cat_list.items);
                df.columns[ci] = .{ .categorical = .{
                    .categories = category_names,
                    .indices = indices,
                } };
            },
        }
    }

    return df;
}

/// Split a DataFrame into N+1 parts at the given row boundaries.
/// E.g. splitDataFrame(df, &[5, 8]) on a 10-row df → 3 DataFrames: rows 0..5, 5..8, 8..10.
/// All returned DataFrames own their data.
pub fn splitDataFrame(allocator: Allocator, df: *const DataFrame, boundaries: []const usize) ![]*DataFrame {
    const n_parts = boundaries.len + 1;
    const num_cols = df.numCols();
    const parts = try allocator.alloc(*DataFrame, n_parts);
    errdefer {
        for (parts) |p| p.deinit();
        allocator.free(parts);
    }

    // Compute start/end row for each part
    var prev: usize = 0;
    for (0..n_parts) |p| {
        const end = if (p < boundaries.len) @min(boundaries[p], df.num_rows) else df.num_rows;
        const n_rows = end - prev;

        const part_df = try DataFrame.init(allocator, n_rows, num_cols);
        parts[p] = part_df;

        // Clone column names
        for (df.col_names, 0..) |name, i| {
            part_df.col_names[i] = try DataFrame.dupeColumnName(allocator, name);
        }

        // Copy column data for this part's row range
        for (df.columns, 0..) |col, ci| {
            switch (col) {
                .numeric => |data| {
                    const d = try DataFrame.allocNumericColumn(allocator, n_rows);
                    for (0..n_rows) |r| d[r] = data[df.rowIndex(prev + r)];
                    part_df.columns[ci] = .{ .numeric = d };
                },
                .categorical => |cat| {
                    const cats = try DataFrame.allocCategoryNames(allocator, cat.categories.len);
                    for (cat.categories, 0..) |s, j| {
                        cats[j] = try DataFrame.dupeCategoryName(allocator, s);
                    }
                    const idx = try DataFrame.allocCategoryIndices(allocator, n_rows);
                    for (0..n_rows) |r| idx[r] = cat.indices[df.rowIndex(prev + r)];
                    part_df.columns[ci] = .{ .categorical = .{ .categories = cats, .indices = idx } };
                },
            }
        }

        prev = end;
    }

    return parts;
}
