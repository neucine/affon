const std = @import("std");
const memory = @import("../../support/compat.zig");

const Allocator = std.mem.Allocator;

pub const DataFrame = struct {
    allocator: Allocator,
    num_rows: usize,
    columns: []Column,
    col_names: [][]const u8,
    /// Row index permutation (for shuffle/sample without moving data)
    index: ?[]u32,
    /// Optional pre-built row-major f64 buffer (all-numeric fast path from CSV parser).
    /// When present and no steps invalidate it, toFlatBuffer can return this directly.
    row_major: ?[]f64,

    pub const Column = union(enum) {
        numeric: []f64,
        categorical: CategoricalColumn,
    };

    pub const CategoricalColumn = struct {
        /// Interned string table: unique category strings
        categories: [][]const u8,
        /// Per-row index into categories
        indices: []u32,
    };

    pub fn allocColumns(allocator: Allocator, num_cols: usize) ![]Column {
        return allocator.alloc(Column, num_cols);
    }

    pub fn freeColumns(allocator: Allocator, columns: []Column) void {
        allocator.free(columns);
    }

    pub fn allocColumnNames(allocator: Allocator, num_cols: usize) ![][]const u8 {
        return allocator.alloc([]const u8, num_cols);
    }

    pub fn freeColumnNames(allocator: Allocator, names: [][]const u8) void {
        allocator.free(names);
    }

    pub fn dupeColumnName(allocator: Allocator, name: []const u8) ![]const u8 {
        const out = try allocator.alloc(u8, name.len);
        @memcpy(out, name);
        return out;
    }

    pub fn freeColumnName(allocator: Allocator, name: []const u8) void {
        allocator.free(name);
    }

    pub fn allocColumnNameFormat(allocator: Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const len = std.fmt.count(fmt, args);
        const out = try allocator.alloc(u8, len);
        _ = try std.fmt.bufPrint(out, fmt, args);
        return out;
    }

    pub fn allocIndex(allocator: Allocator, num_rows: usize) ![]u32 {
        return allocator.alloc(u32, num_rows);
    }

    pub fn freeIndex(allocator: Allocator, index: []u32) void {
        allocator.free(index);
    }

    pub fn allocCategoryNames(allocator: Allocator, count: usize) ![][]const u8 {
        return allocator.alloc([]const u8, count);
    }

    pub fn freeCategoryNames(allocator: Allocator, categories: [][]const u8) void {
        allocator.free(categories);
    }

    pub fn dupeCategoryName(allocator: Allocator, name: []const u8) ![]const u8 {
        const out = try allocator.alloc(u8, name.len);
        @memcpy(out, name);
        return out;
    }

    pub fn freeCategoryName(allocator: Allocator, name: []const u8) void {
        allocator.free(name);
    }

    pub fn allocRowMajor(allocator: Allocator, len: usize) ![]f64 {
        return allocator.alloc(f64, len);
    }

    pub fn freeRowMajor(allocator: Allocator, data: []f64) void {
        allocator.free(data);
    }

    pub fn allocNumericColumn(allocator: Allocator, len: usize) ![]f64 {
        return allocator.alloc(f64, len);
    }

    pub fn freeNumericColumn(allocator: Allocator, data: []f64) void {
        allocator.free(data);
    }

    pub fn allocCategoryIndices(allocator: Allocator, len: usize) ![]u32 {
        return allocator.alloc(u32, len);
    }

    pub fn freeCategoryIndices(allocator: Allocator, data: []u32) void {
        allocator.free(data);
    }

    pub fn init(allocator: Allocator, num_rows: usize, num_cols: usize) !*DataFrame {
        const df = try allocator.create(DataFrame);
        errdefer allocator.destroy(df);

        const columns = try allocColumns(allocator, num_cols);
        errdefer freeColumns(allocator, columns);
        const col_names = try allocColumnNames(allocator, num_cols);
        errdefer freeColumnNames(allocator, col_names);

        df.* = .{
            .allocator = allocator,
            .num_rows = num_rows,
            .columns = columns,
            .col_names = col_names,
            .index = null,
            .row_major = null,
        };
        return df;
    }

    pub fn deinitShallow(self: *DataFrame) void {
        freeColumnNames(self.allocator, self.col_names);
        freeColumns(self.allocator, self.columns);
        self.allocator.destroy(self);
    }

    pub fn deinit(self: *DataFrame) void {
        if (self.row_major) |rm| freeRowMajor(self.allocator, rm);
        if (self.index) |idx| freeIndex(self.allocator, idx);
        for (self.columns) |col| {
            switch (col) {
                .numeric => |data| freeNumericColumn(self.allocator, data),
                .categorical => |cat| {
                    for (cat.categories) |s| freeCategoryName(self.allocator, s);
                    freeCategoryNames(self.allocator, cat.categories);
                    freeCategoryIndices(self.allocator, cat.indices);
                },
            }
        }
        for (self.col_names) |name| freeColumnName(self.allocator, name);
        freeColumnNames(self.allocator, self.col_names);
        freeColumns(self.allocator, self.columns);
        self.allocator.destroy(self);
    }

    pub fn numCols(self: *const DataFrame) usize {
        return self.col_names.len;
    }

    pub fn findColumn(self: *const DataFrame, name: []const u8) ?usize {
        for (self.col_names, 0..) |cn, i| {
            if (std.mem.eql(u8, cn, name)) return i;
        }
        return null;
    }

    /// Ensure the index array exists (identity permutation if not yet created)
    pub fn ensureIndex(self: *DataFrame) !void {
        if (self.index != null) return;
        const idx = try allocIndex(self.allocator, self.num_rows);
        for (idx, 0..) |*v, i| v.* = @intCast(i);
        self.index = idx;
    }

    /// Get the effective row index (respects permutation)
    pub fn rowIndex(self: *const DataFrame, i: usize) usize {
        if (self.index) |idx| return idx[i];
        return i;
    }

    /// Deep clone the DataFrame
    pub fn clone(self: *const DataFrame) !*DataFrame {
        const df = try self.allocator.create(DataFrame);
        errdefer self.allocator.destroy(df);

        const columns = try allocColumns(self.allocator, self.columns.len);
        errdefer freeColumns(self.allocator, columns);
        const col_names = try allocColumnNames(self.allocator, self.col_names.len);
        errdefer freeColumnNames(self.allocator, col_names);

        df.* = .{
            .allocator = self.allocator,
            .num_rows = self.num_rows,
            .columns = columns,
            .col_names = col_names,
            .index = null,
            .row_major = null,
        };

        // Clone column names
        for (self.col_names, 0..) |name, i| {
            df.col_names[i] = try dupeColumnName(self.allocator, name);
        }

        // Clone column data
        for (self.columns, 0..) |col, i| {
            switch (col) {
                .numeric => |data| {
                    const copy = try allocNumericColumn(self.allocator, data.len);
                    @memcpy(copy, data);
                    df.columns[i] = .{ .numeric = copy };
                },
                .categorical => |cat| {
                    const cats = try allocCategoryNames(self.allocator, cat.categories.len);
                    for (cat.categories, 0..) |s, j| {
                        cats[j] = try dupeCategoryName(self.allocator, s);
                    }
                    const indices = try allocCategoryIndices(self.allocator, cat.indices.len);
                    @memcpy(indices, cat.indices);
                    df.columns[i] = .{ .categorical = .{
                        .categories = cats,
                        .indices = indices,
                    } };
                },
            }
        }

        // Clone index
        if (self.index) |idx| {
            const copy = try allocIndex(self.allocator, idx.len);
            @memcpy(copy, idx);
            df.index = copy;
        }

        if (self.row_major) |row_major| {
            const copy = try allocRowMajor(self.allocator, row_major.len);
            @memcpy(copy, row_major);
            df.row_major = copy;
        }

        return df;
    }
};
