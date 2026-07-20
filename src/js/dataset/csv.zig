const std = @import("std");
const DataFrame = @import("DataFrame.zig").DataFrame;
const cfg = @import("../../config.zig");
const compat = @import("../../support/compat.zig");

const Allocator = std.mem.Allocator;
const page_alloc = std.heap.page_allocator;

/// Parse a CSV file into a DataFrame.
/// All-numeric CSVs get a row-major fast path; mixed CSVs build columnar data.
pub fn parseCSV(allocator: Allocator, path: []const u8) !*DataFrame {
    const bytes = try compat.readFileAlloc(page_alloc, path, 512 * 1024 * 1024);
    defer page_alloc.free(bytes);

    var parser = try CSVParser.init(allocator, bytes);
    defer parser.deinit();

    return try parser.parse();
}

const CSVParser = struct {
    allocator: Allocator,
    source: []const u8,
    buf: []u8,
    buf_len: usize,
    buf_pos: usize,
    source_pos: usize,
    eof: bool,
    cell_buf: std.ArrayList(u8),

    fn init(allocator: Allocator, source: []const u8) !CSVParser {
        return CSVParser{
            .allocator = allocator,
            .source = source,
            .buf = try page_alloc.alloc(u8, cfg.config.read().csv.chunk_size.get()),
            .buf_len = 0,
            .buf_pos = 0,
            .source_pos = 0,
            .eof = false,
            .cell_buf = .empty,
        };
    }

    fn deinit(self: *CSVParser) void {
        page_alloc.free(self.buf);
        self.cell_buf.deinit(page_alloc);
    }

    fn fillBuffer(self: *CSVParser) !void {
        if (self.eof) return;
        const remaining = self.source[self.source_pos..];
        const n = @min(remaining.len, self.buf.len);
        if (n == 0) {
            self.eof = true;
            return;
        }
        @memcpy(self.buf[0..n], remaining[0..n]);
        self.source_pos += n;
        self.buf_len = n;
        self.buf_pos = 0;
    }

    fn nextByte(self: *CSVParser) !?u8 {
        if (self.buf_pos >= self.buf_len) {
            try self.fillBuffer();
            if (self.eof) return null;
        }
        const b = self.buf[self.buf_pos];
        self.buf_pos += 1;
        return b;
    }

    /// Parse a single cell into cell_buf. Returns true if line is done (newline/EOF).
    fn parseCell(self: *CSVParser) !bool {
        self.cell_buf.clearRetainingCapacity();
        var in_quotes = false;

        while (true) {
            const maybe_b = try self.nextByte();
            if (maybe_b == null) return true;

            const b = maybe_b.?;

            if (in_quotes) {
                if (b == '"') {
                    const next = try self.nextByte();
                    if (next != null and next.? == '"') {
                        try self.cell_buf.append(page_alloc, '"');
                    } else {
                        in_quotes = false;
                        if (next != null) {
                            if (self.buf_pos > 0) {
                                self.buf_pos -= 1;
                            }
                        }
                    }
                } else {
                    try self.cell_buf.append(page_alloc, b);
                }
            } else if (b == '"') {
                in_quotes = true;
            } else if (b == ',') {
                return false;
            } else if (b == '\n') {
                return true;
            } else if (b == '\r') {
                continue;
            } else {
                try self.cell_buf.append(page_alloc, b);
            }
        }
    }

    fn parse(self: *CSVParser) !*DataFrame {
        const allocator = self.allocator;

        // Parse header
        var header_cells = std.ArrayList([]u8).empty;
        defer {
            for (header_cells.items) |cell| page_alloc.free(cell);
            header_cells.deinit(page_alloc);
        }
        try self.parseLineIntoCells(&header_cells);

        const num_cols = header_cells.items.len;
        if (num_cols == 0) return error.EmptyCSV;

        // Skip leading empty lines before first data row
        try self.skipEmptyLines();
        if (self.eof) {
            // Header only, no data
            const df = try DataFrame.init(allocator, 0, num_cols);
            for (header_cells.items, 0..) |cell, i| {
                df.col_names[i] = try DataFrame.dupeColumnName(allocator, cell);
            }
            return df;
        }

        // Probe first row to detect if all numeric
        var first_row_cells = std.ArrayList([]u8).empty;
        defer {
            for (first_row_cells.items) |cell| page_alloc.free(cell);
            first_row_cells.deinit(page_alloc);
        }
        try self.parseLineIntoCells(&first_row_cells);
        if (first_row_cells.items.len == 0) return error.EmptyCSV;

        // Try parsing all first-row cells as f64
        var first_values = try page_alloc.alloc(f64, num_cols);
        defer page_alloc.free(first_values);
        var all_numeric = true;

        for (first_row_cells.items, 0..) |cell, i| {
            const trimmed = std.mem.trim(u8, cell, " \t");
            if (trimmed.len == 0) {
                first_values[i] = std.math.nan(f64);
                continue;
            }
            first_values[i] = std.fmt.parseFloat(f64, trimmed) catch {
                all_numeric = false;
                break;
            };
        }

        if (all_numeric) {
            return try self.parseNumeric(allocator, &header_cells, num_cols, first_values);
        } else {
            return try self.parseMixed(allocator, &header_cells, num_cols, &first_row_cells);
        }
    }

    /// All-numeric CSV: parse directly into row-major f64 buffer + columnar views.
    fn parseNumeric(
        self: *CSVParser,
        allocator: Allocator,
        header_cells: *std.ArrayList([]u8),
        num_cols: usize,
        first_values: []const f64,
    ) !*DataFrame {
        var data_buf = std.ArrayList(f64).empty;
        defer data_buf.deinit(allocator);
        try data_buf.ensureTotalCapacity(allocator, 1000 * num_cols);

        // Add first row
        try data_buf.appendSlice(allocator, first_values);
        var num_rows: usize = 1;

        // Parse remaining rows directly as f64
        while (true) {
            if (self.buf_pos >= self.buf_len and self.eof) break;
            if (self.buf_pos >= self.buf_len) {
                try self.fillBuffer();
                if (self.eof) break;
            }
            if (self.buf[self.buf_pos] == '\n' or self.buf[self.buf_pos] == '\r') {
                self.buf_pos += 1;
                continue;
            }

            var col: usize = 0;
            while (col < num_cols) {
                const line_done = try self.parseCell();
                const trimmed = std.mem.trim(u8, self.cell_buf.items, " \t");
                const val = if (trimmed.len == 0)
                    std.math.nan(f64)
                else
                    std.fmt.parseFloat(f64, trimmed) catch return error.InvalidNumericValue;
                try data_buf.append(allocator, val);
                col += 1;
                if (line_done) break;
            }
            num_rows += 1;
        }

        // Build DataFrame with row_major fast path
        const df = try DataFrame.init(allocator, num_rows, num_cols);
        for (header_cells.items, 0..) |cell, i| {
            df.col_names[i] = try DataFrame.dupeColumnName(allocator, cell);
        }

        // Store row-major data
        df.row_major = try DataFrame.allocRowMajor(allocator, data_buf.items.len);
        @memcpy(df.row_major.?, data_buf.items);

        // Also set up columnar views (slices into row-major data for pipeline ops)
        // For the fast path, we build per-column numeric arrays
        for (0..num_cols) |col_idx| {
            const col_data = try DataFrame.allocNumericColumn(allocator, num_rows);
            const rm = df.row_major.?;
            for (0..num_rows) |row| {
                col_data[row] = rm[row * num_cols + col_idx];
            }
            df.columns[col_idx] = .{ .numeric = col_data };
        }

        return df;
    }

    /// Mixed types (numeric + categorical): collect cells as strings, then build columnar data.
    fn parseMixed(
        self: *CSVParser,
        allocator: Allocator,
        header_cells: *std.ArrayList([]u8),
        num_cols: usize,
        first_row_cells: *std.ArrayList([]u8),
    ) !*DataFrame {
        // Accumulate raw cell bytes per column
        var col_cells = try allocator.alloc(std.ArrayList([]u8), num_cols);
        defer {
            for (col_cells) |*list| {
                for (list.items) |cell| allocator.free(cell);
                list.deinit(allocator);
            }
            allocator.free(col_cells);
        }
        for (col_cells) |*list| list.* = std.ArrayList([]u8).empty;

        // Add first row
        for (first_row_cells.items, 0..) |cell, col_idx| {
            const trimmed = std.mem.trim(u8, cell, " \t");
            try col_cells[col_idx].append(allocator, try allocator.dupe(u8, trimmed));
        }
        var num_rows: usize = 1;

        // Parse remaining rows
        while (true) {
            if (self.buf_pos >= self.buf_len and self.eof) break;
            if (self.buf_pos >= self.buf_len) {
                try self.fillBuffer();
                if (self.eof) break;
            }
            if (self.buf[self.buf_pos] == '\n' or self.buf[self.buf_pos] == '\r') {
                self.buf_pos += 1;
                continue;
            }

            var col: usize = 0;
            while (col < num_cols) {
                const line_done = try self.parseCell();
                const trimmed = std.mem.trim(u8, self.cell_buf.items, " \t");
                try col_cells[col].append(allocator, try allocator.dupe(u8, trimmed));
                col += 1;
                if (line_done) break;
            }
            while (col < num_cols) {
                try col_cells[col].append(allocator, try allocator.alloc(u8, 0));
                col += 1;
            }
            num_rows += 1;
        }

        // Build DataFrame
        const df = try DataFrame.init(allocator, num_rows, num_cols);
        errdefer df.deinit();

        for (header_cells.items, 0..) |cell, i| {
            df.col_names[i] = try DataFrame.dupeColumnName(allocator, cell);
        }

        for (col_cells, 0..) |cells, col_idx| {
            if (isNumericColumn(cells.items)) {
                const data = try DataFrame.allocNumericColumn(allocator, num_rows);
                for (cells.items, 0..) |cell, row| {
                    data[row] = if (cell.len == 0)
                        std.math.nan(f64)
                    else
                        std.fmt.parseFloat(f64, cell) catch return error.InvalidNumericValue;
                }
                df.columns[col_idx] = .{ .numeric = data };
            } else {
                var intern_map = std.StringHashMap(u32).init(allocator);
                defer intern_map.deinit();
                var categories = std.ArrayList([]const u8).empty;
                defer categories.deinit(allocator);
                const indices = try DataFrame.allocCategoryIndices(allocator, num_rows);

                for (cells.items, 0..) |cell, row| {
                    const gop = try intern_map.getOrPut(cell);
                    if (!gop.found_existing) {
                        gop.value_ptr.* = @intCast(categories.items.len);
                        try categories.append(allocator, try DataFrame.dupeCategoryName(allocator, cell));
                    }
                    indices[row] = gop.value_ptr.*;
                }
                const category_names = try DataFrame.allocCategoryNames(allocator, categories.items.len);
                @memcpy(category_names, categories.items);
                df.columns[col_idx] = .{ .categorical = .{
                    .categories = category_names,
                    .indices = indices,
                } };
            }
        }

        return df;
    }

    fn skipEmptyLines(self: *CSVParser) !void {
        while (true) {
            if (self.buf_pos >= self.buf_len and self.eof) break;
            if (self.buf_pos >= self.buf_len) {
                try self.fillBuffer();
                if (self.eof) break;
            }
            if (self.buf[self.buf_pos] == '\n' or self.buf[self.buf_pos] == '\r') {
                self.buf_pos += 1;
                continue;
            }
            break;
        }
    }

    fn parseLineIntoCells(self: *CSVParser, cells: *std.ArrayList([]u8)) !void {
        while (true) {
            const line_done = try self.parseCell();
            const trimmed = std.mem.trim(u8, self.cell_buf.items, " \t");
            const cell_copy = try page_alloc.dupe(u8, trimmed);
            try cells.append(page_alloc, cell_copy);
            if (line_done) break;
        }
    }
};

fn isNumericColumn(cells: []const []const u8) bool {
    for (cells) |cell| {
        if (cell.len == 0) continue; // empty → NaN
        _ = std.fmt.parseFloat(f64, cell) catch return false;
    }
    return true;
}
