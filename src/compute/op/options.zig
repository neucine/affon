pub const BinaryOptions = struct {};

pub const UnaryOptions = struct {};

pub const ClampOptions = struct {
    min: f64,
    max: f64,
};

pub const MaskedFillOptions = struct {
    value: f64,
};

pub const ReduceAllOptions = struct {
    keepdim: bool = false,
};

pub const ReduceAxisOptions = struct {
    axis: usize,
    keepdim: bool = false,
};

pub const ReshapeOptions = struct {
    shape: []const usize,
};

pub const ConcatOptions = struct {
    axis: usize = 0,
};

pub const StackOptions = struct {
    axis: usize = 0,
};

pub const SliceRange = struct {
    start: usize,
    stop: usize,
    step: isize = 1,
};

pub const SliceOptions = struct {
    ranges: []const SliceRange,
};

pub const GatherOptions = struct {
    axis: usize,
};

pub const EmbeddingOptions = struct {};

pub const IndexSelectOptions = struct {
    axis: usize,
};

pub const ScatterAddOptions = struct {
    axis: usize,
};

pub const TopKOptions = struct {
    k: usize,
    axis: usize,
    largest: bool = true,
    sorted: bool = true,
};

pub const SoftmaxOptions = struct {
    axis: usize,
};

pub const LogSoftmaxOptions = struct {
    axis: usize,
};

pub const LogSoftmaxNllOptions = struct {
    axis: usize = 1,
};
pub const CrossEntropyIndexedOptions = struct {
    axis: usize = 1,
};
pub const CrossEntropyIndexedBackwardOptions = struct {
    axis: usize = 1,
};
pub const CrossEntropyOptions = struct {
    axis: usize = 1,
};

pub const CastOptions = struct {
    to: @import("../tensor/dtype.zig").DType,
};

pub const LayerNormOptions = struct {
    axis: usize,
    eps: f64,
};

pub const RmsNormOptions = struct {
    axis: usize,
    eps: f64,
};

pub const OneHotOptions = struct {
    num_classes: usize,
};

pub const TransposeOptions = struct {
    permutation: ?[]const usize = null,
};

pub const PermuteOptions = struct {
    axes: []const usize,
};

pub const SqueezeOptions = struct {
    axis: ?usize = null,
};

pub const UnsqueezeOptions = struct {
    axis: usize,
};

pub const ReduceToShapeOptions = struct {
    shape: []const usize,
};

pub const OpOptions = union(enum) {
    none: void,
    binary: BinaryOptions,
    unary: UnaryOptions,
    clamp: ClampOptions,
    masked_fill: MaskedFillOptions,
    reduce_all: ReduceAllOptions,
    reduce_axis: ReduceAxisOptions,
    concat: ConcatOptions,
    stack: StackOptions,
    reshape: ReshapeOptions,
    slice: SliceOptions,
    gather: GatherOptions,
    embedding: EmbeddingOptions,
    index_select: IndexSelectOptions,
    scatter_add: ScatterAddOptions,
    topk: TopKOptions,
    one_hot: OneHotOptions,
    cast: CastOptions,
    softmax: SoftmaxOptions,
    log_softmax: LogSoftmaxOptions,
    log_softmax_nll: LogSoftmaxNllOptions,
    cross_entropy_indexed: CrossEntropyIndexedOptions,
    cross_entropy_indexed_backward: CrossEntropyIndexedBackwardOptions,
    cross_entropy: CrossEntropyOptions,
    layer_norm: LayerNormOptions,
    rms_norm: RmsNormOptions,
    transpose: TransposeOptions,
    permute: PermuteOptions,
    squeeze: SqueezeOptions,
    unsqueeze: UnsqueezeOptions,
    reduce_to_shape: ReduceToShapeOptions,
};
