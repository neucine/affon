/** Tested subset, not a claim of general ONNX compatibility. */
export const capabilities = Object.freeze({
  format: 'affon-onnx-static/v1',
  opset: 17,
  conversion_lowerings: Object.freeze({Unsqueeze: 'constant i64 axes to static Reshape'}),
  runtime_dtype: 'f32',
  shapes: 'static-positive',
  artifact: 'offline-converted graph.json + weights.safetensors',
  validated_devices: Object.freeze(['cpu', 'metal'] as const),
  operators: Object.freeze(['Slice','Conv','Reshape','Transpose','Concat','Add','Mul','Div','MatMul','LayerNormalization','Softmax','Cast','Erf','Gather','Gemm','Identity','Pad','Clip','GlobalAveragePool','Flatten'] as const),
  restrictions: Object.freeze({
    Slice: 'constant i64 bounds/axes, positive unit steps, nonempty output',
    Conv: 'NCW 1D or NCHW 2D, constant weights/optional bias, explicit nonnegative pads; strides, dilation and groups supported',
    Pad: 'NCHW spatial, constant zero fill, nonnegative pads',
    Clip: 'finite constant scalar bounds',
    GlobalAveragePool: 'NCHW',
    LayerNormalization: 'final axis, f32, affine, one output',
    Gather: 'constant scalar i64 index',
    Cast: 'f32 identity at runtime',
    Reshape: 'constant shape, allowzero=0',
    MatMul: 'rank two or higher',
    Erf: 'A&S approximation',
  }),
})
