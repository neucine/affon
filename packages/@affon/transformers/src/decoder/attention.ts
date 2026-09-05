import nn from "affon:nn";
import {
  add,
  axes,
  contiguous,
  module as computeModule,
  masked_fill,
  matmul,
  mul,
  permute,
  reshape,
  softmax,
  tensor,
} from "affon:compute";
import type { Tensor } from "affon:compute";

import { causal_mask } from "../sequence.ts";

function graphSafeDevice(value: { device: "cpu" | "metal" | "cuda" | `cuda:${number}` }): "cpu" | "metal" | "cuda" | `cuda:${number}` | undefined {
  try {
    return value.device;
  } catch {
    return undefined;
  }
}

function graphCaptureDevice(
  activation: Tensor<number[], "f32">,
  fallback: Tensor<number[], "f32">,
): "cpu" | "metal" | "cuda" | `cuda:${number}` | undefined {
  return graphSafeDevice(activation) ?? graphSafeDevice(fallback);
}

export interface SelfAttentionOptions {
  numHeads: number;
  causal?: boolean;
}

export type SelfAttentionModule = nn.Module<[Tensor<number[], "f32">], Tensor<number[], "f32">> & {
  q_proj: nn.LinearLayer<number, number, "f32">;
  k_proj: nn.LinearLayer<number, number, "f32">;
  v_proj: nn.LinearLayer<number, number, "f32">;
  out_proj: nn.LinearLayer<number, number, "f32">;
};

function project3d(
  proj: nn.LinearLayer<number, number, "f32">,
  x: Tensor<[number, number, number], "f32">,
  outDim: number,
): Tensor<number[], "f32"> {
  const projected = matmul(
    x,
    proj.weight as Tensor<[number, number], "f32">,
    { hint: "projection", source: "higher_level_module" },
  ) as Tensor<number[], "f32">;
  nn.diagnostics.assert("finite", { path: "SelfAttention.project3d.linear", value: projected });
  const out = contiguous(
    add(
      projected,
      proj.bias as Tensor<[1, number], "f32">,
    ),
  ) as Tensor<number[], "f32">;
  nn.diagnostics.assert("finite", { path: "SelfAttention.project3d.reshaped", value: out });
  return out;
}

export function SelfAttention(
  dModel: number,
  opts: SelfAttentionOptions,
): SelfAttentionModule {
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new AffonError(
      "invalid_arg",
      "SelfAttention dModel must be a positive integer",
    );
  }
  if (!Number.isInteger(opts.numHeads) || opts.numHeads <= 0) {
    throw new AffonError(
      "invalid_arg",
      "SelfAttention numHeads must be a positive integer",
    );
  }
  if (dModel % opts.numHeads !== 0) {
    throw new AffonError(
      "invalid_arg",
      "SelfAttention dModel must be divisible by numHeads",
    );
  }

  const numHeads = opts.numHeads;
  const headDim = dModel / numHeads;
  const scale = 1 / Math.sqrt(headDim);
  const causal = opts.causal ?? true;

  const q_proj = nn.Linear<number, number, "f32">(dModel, dModel, {
    dtype: "f32",
  });
  const k_proj = nn.Linear<number, number, "f32">(dModel, dModel, {
    dtype: "f32",
  });
  const v_proj = nn.Linear<number, number, "f32">(dModel, dModel, {
    dtype: "f32",
  });
  const out_proj = nn.Linear<number, number, "f32">(dModel, dModel, {
    dtype: "f32",
  });

	  return computeModule({
	    q_proj,
	    k_proj,
	    v_proj,
	    out_proj,
	  } as any, function (_state, x: Tensor<number[], "f32">): Tensor<number[], "f32"> {
      if (x.ndim !== 3) {
        throw new AffonError(
          "invalid_shape",
          "SelfAttention expects input shaped [batch, seq, d_model]",
        );
      }
      if (x.shape[2] !== dModel) {
        throw new AffonError(
          "shape_mismatch",
          "SelfAttention input last dimension must equal dModel",
        );
      }

      const batch = x.shape[0] as number;
      const seq = x.shape[1] as number;
      const x3 = x as Tensor<[number, number, number], "f32">;

      let q_projOut: Tensor<number[], "f32"> | null = project3d(
        q_proj,
        x3,
        dModel,
      );
      let qReshaped: Tensor<number[], "f32"> | null = reshape(q_projOut, [
        batch,
        seq,
        numHeads,
        headDim,
      ], { axes: [axes.batch, axes.token, axes.head, axes.feature] });
      q_projOut = null;
      const q = permute(qReshaped, [0, 2, 1, 3]);
      qReshaped = null;
      nn.diagnostics.assert("finite", { path: "SelfAttention.q", value: q });

      let k_projOut: Tensor<number[], "f32"> | null = project3d(
        k_proj,
        x3,
        dModel,
      );
      let kReshaped: Tensor<number[], "f32"> | null = reshape(k_projOut, [
        batch,
        seq,
        numHeads,
        headDim,
      ], { axes: [axes.batch, axes.token, axes.head, axes.feature] });
      k_projOut = null;
      const k = permute(kReshaped, [0, 2, 1, 3]);
      kReshaped = null;
      nn.diagnostics.assert("finite", { path: "SelfAttention.k", value: k });

      let v_projOut: Tensor<number[], "f32"> | null = project3d(
        v_proj,
        x3,
        dModel,
      );
      let vReshaped: Tensor<number[], "f32"> | null = reshape(v_projOut, [
        batch,
        seq,
        numHeads,
        headDim,
      ], { axes: [axes.batch, axes.token, axes.head, axes.feature] });
      v_projOut = null;
      const v = permute(vReshaped, [0, 2, 1, 3]);
      vReshaped = null;
      nn.diagnostics.assert("finite", { path: "SelfAttention.v", value: v });

      const device = graphCaptureDevice(
        x as Tensor<number[], "f32">,
        q_proj.weight as Tensor<number[], "f32">,
      );
      const scaleTensor = tensor([scale], device
        ? { dtype: x.dtype, device }
        : { dtype: x.dtype }) as Tensor<[1], "f32">;
      const mask = causal
        ? (causal_mask(seq, device ? { dtype: x.dtype, device } : { dtype: x.dtype }) as Tensor<
            [number, number],
            "f32"
          >)
        : null;
      const kt = permute(k, [0, 1, 3, 2]);
      const rawLogits = matmul(q, kt, { hint: "attention_scores", source: "higher_level_module" });
      let logits = mul(rawLogits, scaleTensor);
      if (mask) {
        logits = masked_fill(logits, mask, -1e9) as Tensor<number[], "f32">;
      }
      const weights = softmax(logits, 3);
      const context = matmul(weights, v, { hint: "attention_values", source: "higher_level_module" });
      const permuted = permute(context, [0, 2, 1, 3]);
      const merged = reshape(contiguous(permuted), [batch, seq, dModel], { axes: [axes.batch, axes.token, axes.feature] }) as Tensor<
        number[],
        "f32"
      >;
      const out = project3d(
        out_proj,
        merged as Tensor<[number, number, number], "f32">,
        dModel,
      );
      nn.diagnostics.assert("finite", { path: "SelfAttention.out_proj", value: out });
      return out;
  }) as SelfAttentionModule;
}
