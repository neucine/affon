// Generic static NCHW spatial lowerings. No model names or layer structure.
import {
  add,
  contiguous,
  index_select,
  matmul,
  mul,
  permute,
  reshape,
  sum,
  tensor,
  transpose,
  where,
} from "affon:compute";
import type { Device, Tensor } from "affon:compute";

type Conv = {
  kernel: number[];
  strides?: number[];
  dilations?: number[];
  pads?: number[];
  group?: number;
};

// Indices address spatial positions only and are reused across batches/channels.
// Out-of-image positions gather index zero and are then masked to zero.
function spatial_gather(
  height: number,
  width: number,
  oh: number,
  ow: number,
  kh: number,
  kw: number,
  strides: number[],
  dilations: number[],
  pads: number[],
  device: Device,
) {
  const indices: number[] = [],
    mask: number[] = [];
  let padded = false;
  for (let y = 0; y < oh; y++)
    for (let x = 0; x < ow; x++)
      for (let ky = 0; ky < kh; ky++)
        for (let kx = 0; kx < kw; kx++) {
          const iy = y * strides[0] + ky * dilations[0] - pads[0],
            ix = x * strides[1] + kx * dilations[1] - pads[1];
          const valid = iy >= 0 && iy < height && ix >= 0 && ix < width;
          indices.push(valid ? iy * width + ix : 0);
          mask.push(valid ? 1 : 0);
          padded ||= !valid;
        }
  const index = tensor(indices, { dtype: "i64", device });
  const valid = padded ? tensor(mask, { dtype: "f32", device }) : null;
  const zero = valid ? tensor(0, { dtype: "f32", device }) : null;
  return (input: Tensor) => {
    const [batch, channels] = input.shape;
    const selected = index_select(
      reshape(contiguous(input), [batch, channels, height * width]),
      2,
      index,
    );
    return valid ? where(valid, selected, zero!) : selected;
  };
}

export function prepare_pad(shape: number[], pads: number[], device: Device) {
  const [batch, channels, height, width] = shape;
  if (pads.every((x) => x === 0)) return (x: Tensor) => x;
  const oh = height + pads[0] + pads[2],
    ow = width + pads[1] + pads[3];
  const gather = spatial_gather(
    height,
    width,
    oh,
    ow,
    1,
    1,
    [1, 1],
    [1, 1],
    pads,
    device,
  );
  return (x: Tensor) => reshape(gather(x), [batch, channels, oh, ow]);
}

export function prepare_conv(
  shape: number[],
  weight: Tensor,
  bias: Tensor | undefined,
  a: Conv,
  device: Device,
) {
  const [batch, channels, height, width] = shape,
    [outputs, cg, kh, kw] = weight.shape;
  // Legacy v1 manifests only had kernel; their convolution was nonoverlapping.
  const strides = a.strides ?? a.kernel,
    dilations = a.dilations ?? [1, 1],
    pads = a.pads ?? [0, 0, 0, 0],
    groups = a.group ?? 1;
  const oh =
    Math.floor(
      (height + pads[0] + pads[2] - dilations[0] * (kh - 1) - 1) / strides[0],
    ) + 1;
  const ow =
    Math.floor(
      (width + pads[1] + pads[3] - dilations[1] * (kw - 1) - 1) / strides[1],
    ) + 1;
  if (
    groups < 1 ||
    channels !== cg * groups ||
    outputs % groups ||
    oh < 1 ||
    ow < 1
  )
    throw Error("Invalid convolution dimensions");
  const spatial = oh * ow,
    kernel = kh * kw;
  const affine = (x: Tensor) =>
    bias ? add(x, reshape(bias, [1, outputs, 1, 1])) : x;
  if (
    groups === 1 &&
    pads.every((x) => x === 0) &&
    dilations.every((x) => x === 1) &&
    strides[0] === kh &&
    strides[1] === kw &&
    height % kh === 0 &&
    width % kw === 0
  ) {
    const matrix = contiguous(
      transpose(reshape(weight, [outputs, channels * kernel]), 0, 1),
    );
    return (x: Tensor) => {
      const blocks = reshape(contiguous(x), [batch, channels, oh, kh, ow, kw]);
      const patches = reshape(contiguous(permute(blocks, [0, 2, 4, 1, 3, 5])), [
        batch,
        spatial,
        channels * kernel,
      ]);
      let projected = matmul(patches, matrix);
      if (bias) projected = add(projected, bias);
      return permute(
        reshape(projected, [batch, oh, ow, outputs]),
        [0, 3, 1, 2],
      );
    };
  }
  const gather = spatial_gather(
    height,
    width,
    oh,
    ow,
    kh,
    kw,
    strides,
    dilations,
    pads,
    device,
  );
  if (groups === channels && outputs === channels) {
    const kernels = reshape(weight, [1, channels, 1, kernel]);
    return (x: Tensor) =>
      affine(
        reshape(
          sum(
            mul(
              reshape(gather(x), [batch, channels, spatial, kernel]),
              kernels,
            ),
            3,
          ),
          [batch, channels, oh, ow],
        ),
      );
  }
  const og = outputs / groups;
  const matrices = contiguous(
    transpose(reshape(weight, [groups, og, cg * kernel]), 1, 2),
  );
  return (x: Tensor) => {
    const patches = reshape(gather(x), [batch, groups, cg, spatial, kernel]);
    const grouped = reshape(contiguous(permute(patches, [1, 0, 3, 2, 4])), [
      groups,
      batch * spatial,
      cg * kernel,
    ]);
    const projected = reshape(matmul(grouped, matrices), [
      groups,
      batch,
      oh,
      ow,
      og,
    ]);
    return affine(
      reshape(contiguous(permute(projected, [1, 0, 4, 2, 3])), [
        batch,
        outputs,
        oh,
        ow,
      ]),
    );
  };
}
