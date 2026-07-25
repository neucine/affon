// Shared serialized captured-program boundary used by TS capture, native
// lowering, and native execution helpers.
pub const CapturedMatmulExecutionJson = struct {
    hint: ?[]const u8 = null,
    source: ?[]const u8 = null,
};

pub const CapturedSliceRangeJson = @import("compute").operation.SliceRange;

pub const CapturedSliceSelectorJson = struct {
    kind: []const u8,
    index: ?i64 = null,
    start: ?i64 = null,
    stop: ?i64 = null,
    step: ?isize = null,
};

pub const CapturedNodeJson = struct {
    id: u32,
    kind: []const u8,
    module_path: ?[]const u8 = null,
    index: ?usize = null,
    input: ?u32 = null,
    mask: ?u32 = null,
    left: ?u32 = null,
    right: ?u32 = null,
    cond: ?u32 = null,
    onTrue: ?u32 = null,
    onFalse: ?u32 = null,
    logits: ?u32 = null,
    targets: ?u32 = null,
    axis: ?usize = null,
    dim: ?usize = null,
    k: ?usize = null,
    keepdim: ?bool = null,
    shape: ?[]usize = null,
    axes: ?[]usize = null,
    inputs: ?[]u32 = null,
    slice_selectors: ?[]CapturedSliceSelectorJson = null,
    slice_ranges: ?[]CapturedSliceRangeJson = null,
    data: ?f64 = null,
    value: ?f64 = null,
    min: ?f64 = null,
    max: ?f64 = null,
    dtype: ?[]const u8 = null,
    numClasses: ?usize = null,
    execution: ?CapturedMatmulExecutionJson = null,
};

pub const CapturedProgramJson = struct {
    inputArity: usize,
    boundInputCount: usize,
    inputCount: usize,
    outputId: u32,
    nodes: []CapturedNodeJson,
};
