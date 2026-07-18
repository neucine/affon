pub const DType = enum {
    f32,
    f64,
    i64,

    pub fn size(self: DType) usize {
        return switch (self) {
            .f32 => @sizeOf(f32),
            .f64 => @sizeOf(f64),
            .i64 => @sizeOf(i64),
        };
    }
};
