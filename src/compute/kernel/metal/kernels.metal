#include <metal_stdlib>
using namespace metal;

kernel void affon_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] + b[id];
}

kernel void affon_sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] - b[id];
}

kernel void affon_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] * b[id];
}

kernel void affon_div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] / b[id];
}

kernel void affon_eq_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] == b[id] ? 1 : 0;
}

kernel void affon_lt_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] < b[id] ? 1 : 0;
}

kernel void affon_gt_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] > b[id] ? 1 : 0;
}

kernel void affon_add_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] + b[id];
}

kernel void affon_sub_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] - b[id];
}

kernel void affon_mul_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] * b[id];
}

kernel void affon_div_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] / b[id];
}

kernel void affon_eq_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] == b[id] ? 1 : 0;
}

kernel void affon_lt_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] < b[id] ? 1 : 0;
}

kernel void affon_gt_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] > b[id] ? 1 : 0;
}

kernel void affon_fill_f32(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = value;
}

kernel void affon_dot_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float acc = 0.0f;
  for (uint i = 0; i < len; ++i) {
    acc += a[i] * b[i];
  }
  out[0] = acc;
}

kernel void affon_dot_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long acc = 0;
  for (uint i = 0; i < len; ++i) {
    acc += a[i] * b[i];
  }
  out[0] = acc;
}


kernel void affon_adam_step_f32(
    device float* param [[buffer(0)]],
    device float* m [[buffer(1)]],
    device float* v [[buffer(2)]],
    device const float* grad [[buffer(3)]],
    constant float& beta1 [[buffer(4)]],
    constant float& beta2 [[buffer(5)]],
    constant float& bias_correction1 [[buffer(6)]],
    constant float& bias_correction2 [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& lr [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    constant uint& len [[buffer(11)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;

  float p = param[id];
  const float g = grad[id];

  if (weight_decay != 0.0f) {
    p -= lr * weight_decay * p;
  }

  const float next_m = beta1 * m[id] + (1.0f - beta1) * g;
  const float next_v = beta2 * v[id] + (1.0f - beta2) * g * g;
  m[id] = next_m;
  v[id] = next_v;

  const float m_hat = next_m / bias_correction1;
  const float v_hat = next_v / bias_correction2;
  param[id] = p - lr * (m_hat / (sqrt(v_hat) + eps));
}

kernel void affon_contiguous_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& offset [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += coord * strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_contiguous_i64(
    device const long* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& offset [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += coord * strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_contiguous_signed_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant int* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant int& offset [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  int input_off = offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += int(coord) * strides[d];
  }
  out[id] = input[uint(input_off)];
}

kernel void affon_contiguous_signed_i64(
    device const long* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant int* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant int& offset [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  int input_off = offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += int(coord) * strides[d];
  }
  out[id] = input[uint(input_off)];
}

kernel void affon_add_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] + b[b_off];
}

kernel void affon_sub_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] - b[b_off];
}

kernel void affon_mul_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] * b[b_off];
}

kernel void affon_div_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] / b[b_off];
}

kernel void affon_eq_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] == b[b_off] ? 1 : 0;
}

kernel void affon_lt_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] < b[b_off] ? 1 : 0;
}

kernel void affon_gt_broadcast_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* a_strides [[buffer(4)]],
    constant uint* b_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& a_offset [[buffer(7)]],
    constant uint& b_offset [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = a[a_off] > b[b_off] ? 1 : 0;
}

kernel void affon_exp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = exp(a[id]);
}

kernel void affon_log_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = log(a[id]);
}

kernel void affon_sqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = sqrt(a[id]);
}

kernel void affon_abs_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = fabs(a[id]);
}

kernel void affon_abs_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const long x = a[id];
  out[id] = x < 0 ? -x : x;
}

kernel void affon_sign_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = a[id] > 0.0f ? 1.0f : (a[id] < 0.0f ? -1.0f : 0.0f);
}

kernel void affon_sign_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const long x = a[id];
  out[id] = x > 0 ? 1 : (x < 0 ? -1 : 0);
}

kernel void affon_gelu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const float x = a[id];
  if (x >= 10.0f) {
    out[id] = x;
    return;
  }
  if (x <= -10.0f) {
    out[id] = 0.0f;
    return;
  }
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  out[id] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void affon_gelu_grad_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const float x = a[id];
  if (x >= 10.0f) {
    out[id] = 1.0f;
    return;
  }
  if (x <= -10.0f) {
    out[id] = 0.0f;
    return;
  }
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float cubic_coeff = 0.044715f;
  const float inner_cubic_coeff = 0.134145f;
  const float inner = sqrt_2_over_pi * (x + cubic_coeff * x * x * x);
  const float tanh_inner = tanh(inner);
  const float sech2 = 1.0f - tanh_inner * tanh_inner;
  const float term = sqrt_2_over_pi * (1.0f + inner_cubic_coeff * x * x);
  out[id] = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * term;
}

kernel void affon_neg_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = -a[id];
}

kernel void affon_neg_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = -a[id];
}

kernel void affon_relu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = max(a[id], 0.0f);
}

kernel void affon_relu_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = max(a[id], (long)0);
}

kernel void affon_sigmoid_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = 1.0f / (1.0f + exp(-a[id]));
}

kernel void affon_silu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const float x = a[id];
  const float sig = 1.0f / (1.0f + exp(-x));
  out[id] = x * sig;
}

kernel void affon_tanh_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  const float x = a[id];
  if (x >= 10.0f) {
    out[id] = 1.0f;
  } else if (x <= -10.0f) {
    out[id] = -1.0f;
  } else {
    out[id] = tanh(x);
  }
}

kernel void affon_clamp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& lo [[buffer(2)]],
    constant float& hi [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = clamp(a[id], lo, hi);
}

kernel void affon_clamp_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant long& lo [[buffer(2)]],
    constant long& hi [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = clamp(a[id], lo, hi);
}

kernel void affon_clamp_grad_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& lo [[buffer(2)]],
    constant float& hi [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  const float x = a[id];
  out[id] = (x > lo && x < hi) ? 1.0f : 0.0f;
}

kernel void affon_softmax_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float max_val = a[id];
    for (uint r = 1; r < rows; ++r) {
      max_val = max(max_val, a[r * cols + id]);
    }
    float sum_exp = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      const uint idx = r * cols + id;
      const float v = exp(a[idx] - max_val);
      out[idx] = v;
      sum_exp += v;
    }
    for (uint r = 0; r < rows; ++r) {
      const uint idx = r * cols + id;
      out[idx] = out[idx] / sum_exp;
    }
  } else {
    if (id >= rows) return;
    const uint base = id * cols;
    float max_val = a[base];
    for (uint c = 1; c < cols; ++c) {
      max_val = max(max_val, a[base + c]);
    }
    float sum_exp = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      const uint idx = base + c;
      const float v = exp(a[idx] - max_val);
      out[idx] = v;
      sum_exp += v;
    }
    for (uint c = 0; c < cols; ++c) {
      const uint idx = base + c;
      out[idx] = out[idx] / sum_exp;
    }
  }
}

kernel void affon_softmax_nd_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* input_strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& axis_size [[buffer(6)]],
    constant uint& len [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % shape[d];
    rem /= shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float max_val = -INFINITY;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) {
      off += coord[d] * input_strides[d];
    }
    max_val = max(max_val, a[off]);
  }

  float sum_exp = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) {
      off += coord[d] * input_strides[d];
    }
    sum_exp += exp(a[off] - max_val);
  }

  uint in_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) {
    in_off += coord[d] * input_strides[d];
  }
  out[id] = exp(a[in_off] - max_val) / sum_exp;
}

kernel void affon_log_softmax_nll_nd_f32(
    device const float* logits [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& groups [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;

  float acc = 0.0f;
  for (uint g = 0; g < groups; ++g) {
    uint coord[8];
    uint rem = g;
    for (uint rev = 0; rev < ndim; ++rev) {
      const uint d = ndim - 1 - rev;
      if (d == axis) {
        coord[d] = 0;
      } else {
        const uint dim = shape[d];
        coord[d] = rem % dim;
        rem /= dim;
      }
    }

    float max_val = -INFINITY;
    for (uint j = 0; j < axis_size; ++j) {
      coord[axis] = j;
      uint off = 0;
      for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
      max_val = max(max_val, logits[off]);
    }

    float sum_exp = 0.0f;
    for (uint j = 0; j < axis_size; ++j) {
      coord[axis] = j;
      uint off = 0;
      for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
      sum_exp += exp(logits[off] - max_val);
    }
    const float log_sum_exp = log(sum_exp);

    float row_nll = 0.0f;
    for (uint j = 0; j < axis_size; ++j) {
      coord[axis] = j;
      uint off = 0;
      for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
      const float log_softmax = (logits[off] - max_val) - log_sum_exp;
      row_nll += -targets[off] * log_softmax;
    }
    acc += row_nll;
  }
  out[0] = acc / float(groups);
}

kernel void affon_cross_entropy_indexed_f32(
    device const float* logits [[buffer(0)]],
    device const long* targets [[buffer(1)]],
    device float* row_losses [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& classes [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  if (tgid >= rows) return;

  const long target = targets[tgid];
  if (target < 0 || target >= (long)classes) {
    if (lid == 0) {
      atomic_store_explicit(status, 1u, memory_order_relaxed);
      row_losses[tgid] = 0.0f;
    }
    return;
  }

  const uint base = tgid * classes;

  float local_max = -INFINITY;
  for (uint c = lid; c < classes; c += tgs) {
    local_max = max(local_max, logits[base + c]);
  }
  shmem[lid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] = max(shmem[lid], shmem[lid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float row_max = shmem[0];
  float local_sum_exp = 0.0f;
  for (uint c = lid; c < classes; c += tgs) {
    local_sum_exp += exp(logits[base + c] - row_max);
  }
  shmem[lid] = local_sum_exp;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    row_losses[tgid] = log(shmem[0]) + row_max - logits[base + (uint)target];
  }
}

kernel void affon_cross_entropy_indexed_transposed_f32(
    device const float* logits [[buffer(0)]],
    device const long* targets [[buffer(1)]],
    device float* row_losses [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& classes [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  if (tgid >= rows) return;

  const long target = targets[tgid];
  if (target < 0 || target >= (long)classes) {
    if (lid == 0) {
      atomic_store_explicit(status, 1u, memory_order_relaxed);
      row_losses[tgid] = 0.0f;
    }
    return;
  }

  float local_max = -INFINITY;
  for (uint c = lid; c < classes; c += tgs) {
    local_max = max(local_max, logits[c * rows + tgid]);
  }
  shmem[lid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] = max(shmem[lid], shmem[lid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float row_max = shmem[0];
  float local_sum_exp = 0.0f;
  for (uint c = lid; c < classes; c += tgs) {
    local_sum_exp += exp(logits[c * rows + tgid] - row_max);
  }
  shmem[lid] = local_sum_exp;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    row_losses[tgid] = log(shmem[0]) + row_max - logits[(uint)target * rows + tgid];
  }
}

kernel void affon_cross_entropy_indexed_backward_f32(
    device const float* logits [[buffer(0)]],
    device const long* targets [[buffer(1)]],
    device const float* grad_out [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& classes [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  if (tgid >= rows) return;

  const long target = targets[tgid];
  if (target < 0 || target >= (long)classes) {
    if (lid == 0) {
      atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
    return;
  }

  const uint base = tgid * classes;

  float local_max = -INFINITY;
  for (uint c = lid; c < classes; c += tgs) {
    local_max = max(local_max, logits[base + c]);
  }
  shmem[lid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] = max(shmem[lid], shmem[lid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float row_max = shmem[0];
  float local_sum_exp = 0.0f;
  for (uint c = lid; c < classes; c += tgs) {
    local_sum_exp += exp(logits[base + c] - row_max);
  }
  shmem[lid] = local_sum_exp;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rows = grad_out[0] / float(rows);
  const float denom = shmem[0];
  for (uint c = lid; c < classes; c += tgs) {
    float grad = exp(logits[base + c] - row_max) / denom;
    grad *= inv_rows;
    if (c == uint(target)) grad -= inv_rows;
    out[base + c] = grad;
  }
}

kernel void affon_where_f32(
    device const float* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = (cond[id] != 0.0f) ? a[id] : b[id];
}

kernel void affon_where_i64_f32(
    device const long* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = (cond[id] != 0) ? a[id] : b[id];
}

kernel void affon_where_f32_i64(
    device const float* cond [[buffer(0)]],
    device const long* a [[buffer(1)]],
    device const long* b [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = (cond[id] != 0.0f) ? a[id] : b[id];
}

kernel void affon_where_i64_i64(
    device const long* cond [[buffer(0)]],
    device const long* a [[buffer(1)]],
    device const long* b [[buffer(2)]],
    device long* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = (cond[id] != 0) ? a[id] : b[id];
}

kernel void affon_masked_fill_i64_f32(
    device const float* input [[buffer(0)]],
    device const long* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& fill_value [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  out[id] = (mask[id] != 0) ? fill_value : input[id];
}

kernel void affon_masked_fill_i64_i64(
    device const long* input [[buffer(0)]],
    device const long* mask [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant long& fill_value [[buffer(3)]],
    constant uint& len [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  out[id] = (mask[id] != 0) ? fill_value : input[id];
}

kernel void affon_masked_fill_broadcast_i64_f32(
    device const float* input [[buffer(0)]],
    device const long* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint* mask_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& input_offset [[buffer(7)]],
    constant uint& mask_offset [[buffer(8)]],
    constant float& fill_value [[buffer(9)]],
    constant uint& len [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = input_offset;
  uint mask_off = mask_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += coord * input_strides[d];
    mask_off += coord * mask_strides[d];
  }
  out[id] = (mask[mask_off] != 0) ? fill_value : input[input_off];
}

kernel void affon_masked_fill_broadcast_i64_i64(
    device const long* input [[buffer(0)]],
    device const long* mask [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint* mask_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& input_offset [[buffer(7)]],
    constant uint& mask_offset [[buffer(8)]],
    constant long& fill_value [[buffer(9)]],
    constant uint& len [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = input_offset;
  uint mask_off = mask_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    input_off += coord * input_strides[d];
    mask_off += coord * mask_strides[d];
  }
  out[id] = (mask[mask_off] != 0) ? fill_value : input[input_off];
}

kernel void affon_where_broadcast_f32(
    device const float* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* shape [[buffer(4)]],
    constant uint* cond_strides [[buffer(5)]],
    constant uint* a_strides [[buffer(6)]],
    constant uint* b_strides [[buffer(7)]],
    constant uint& ndim [[buffer(8)]],
    constant uint& cond_offset [[buffer(9)]],
    constant uint& a_offset [[buffer(10)]],
    constant uint& b_offset [[buffer(11)]],
    constant uint& len [[buffer(12)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint cond_off = cond_offset;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    cond_off += coord * cond_strides[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = (cond[cond_off] != 0.0f) ? a[a_off] : b[b_off];
}

kernel void affon_where_broadcast_i64_f32(
    device const long* cond [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint* shape [[buffer(4)]],
    constant uint* cond_strides [[buffer(5)]],
    constant uint* a_strides [[buffer(6)]],
    constant uint* b_strides [[buffer(7)]],
    constant uint& ndim [[buffer(8)]],
    constant uint& cond_offset [[buffer(9)]],
    constant uint& a_offset [[buffer(10)]],
    constant uint& b_offset [[buffer(11)]],
    constant uint& len [[buffer(12)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint cond_off = cond_offset;
  uint a_off = a_offset;
  uint b_off = b_offset;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    cond_off += coord * cond_strides[d];
    a_off += coord * a_strides[d];
    b_off += coord * b_strides[d];
  }
  out[id] = (cond[cond_off] != 0) ? a[a_off] : b[b_off];
}

kernel void affon_scatter_add_f32(
    device const float* index [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint* dst_shape [[buffer(3)]],
    constant uint* src_shape [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& src_len [[buffer(7)]],
    constant uint& dst_len [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= src_len) return;
  uint src_coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    src_coord[d] = rem % src_shape[d];
    rem /= src_shape[d];
  }

  const int scatter_idx = int(index[id]);
  if (scatter_idx < 0 || uint(scatter_idx) >= dst_shape[axis]) return;
  uint out_id = 0;
  for (uint d = 0; d < ndim; ++d) {
    const uint coord = (d == axis) ? uint(scatter_idx) : src_coord[d];
    out_id = out_id * dst_shape[d] + coord;
    }
  if (out_id < dst_len) atomic_fetch_add_explicit(&out[out_id], src[id], memory_order_relaxed);
}

kernel void affon_scatter_add_i64_f32(
    device const long* index [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint* dst_shape [[buffer(3)]],
    constant uint* src_shape [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& src_len [[buffer(7)]],
    constant uint& dst_len [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= src_len) return;
  uint src_coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    src_coord[d] = rem % src_shape[d];
    rem /= src_shape[d];
  }

  const long scatter_idx = index[id];
  if (scatter_idx < 0 || uint(scatter_idx) >= dst_shape[axis]) return;
  uint out_id = 0;
  for (uint d = 0; d < ndim; ++d) {
    const uint coord = (d == axis) ? uint(scatter_idx) : src_coord[d];
    out_id = out_id * dst_shape[d] + coord;
    }
  if (out_id < dst_len) atomic_fetch_add_explicit(&out[out_id], src[id], memory_order_relaxed);
}

kernel void affon_reduce_sum_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float acc = 0.0f;
  for (uint i = 0; i < len; ++i) {
    acc += a[i];
  }
  out[0] = acc;
}

kernel void affon_reduce_sum_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long acc = 0;
  for (uint i = 0; i < len; ++i) {
    acc += a[i];
  }
  out[0] = acc;
}

kernel void affon_reduce_mean_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float acc = 0.0f;
  for (uint i = 0; i < len; ++i) {
    acc += a[i];
  }
  out[0] = acc / float(len);
}

kernel void affon_reduce_max_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  for (uint i = 1; i < len; ++i) {
    best = max(best, a[i]);
  }
  out[0] = best;
}

kernel void affon_reduce_max_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long best = a[0];
  for (uint i = 1; i < len; ++i) {
    best = max(best, a[i]);
  }
  out[0] = best;
}

kernel void affon_reduce_min_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  for (uint i = 1; i < len; ++i) {
    best = min(best, a[i]);
  }
  out[0] = best;
}

kernel void affon_reduce_min_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long best = a[0];
  for (uint i = 1; i < len; ++i) {
    best = min(best, a[i]);
  }
  out[0] = best;
}

kernel void affon_reduce_argmax_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] > best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = float(best_idx);
}

kernel void affon_reduce_argmax_i64_f32(
    device const float* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] > best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = ulong(best_idx);
}

kernel void affon_reduce_argmax_i64_i64(
    device const long* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] > best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = ulong(best_idx);
}

kernel void affon_reduce_argmin_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] < best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = float(best_idx);
}

kernel void affon_reduce_argmin_i64_f32(
    device const float* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] < best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = ulong(best_idx);
}

kernel void affon_reduce_argmin_i64_i64(
    device const long* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  long best = a[0];
  uint best_idx = 0;
  for (uint i = 1; i < len; ++i) {
    if (a[i] < best) {
      best = a[i];
      best_idx = i;
    }
  }
  out[0] = ulong(best_idx);
}

kernel void affon_reduce_variance_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float mean = 0.0f;
  for (uint i = 0; i < len; ++i) {
    mean += a[i];
  }
  mean /= float(len);
  float acc = 0.0f;
  for (uint i = 0; i < len; ++i) {
    const float d = a[i] - mean;
    acc += d * d;
  }
  out[0] = acc / float(len);
}

kernel void affon_reduce_std_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  if (id != 0) return;
  float mean = 0.0f;
  for (uint i = 0; i < len; ++i) {
    mean += a[i];
  }
  mean /= float(len);
  float acc = 0.0f;
  for (uint i = 0; i < len; ++i) {
    const float d = a[i] - mean;
    acc += d * d;
  }
  out[0] = sqrt(acc / float(len));
}

kernel void affon_reduce_axis_sum_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      acc += a[r * cols + id];
    }
    out[id] = acc;
  } else {
    if (id >= rows) return;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      acc += a[id * cols + c];
    }
    out[id] = acc;
  }
}

kernel void affon_reduce_axis_sum_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    long acc = 0;
    for (uint r = 0; r < rows; ++r) {
      acc += a[r * cols + id];
    }
    out[id] = acc;
  } else {
    if (id >= rows) return;
    long acc = 0;
    for (uint c = 0; c < cols; ++c) {
      acc += a[id * cols + c];
    }
    out[id] = acc;
  }
}

kernel void affon_reduce_axis_mean_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      acc += a[r * cols + id];
    }
    out[id] = acc / float(rows);
  } else {
    if (id >= rows) return;
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      acc += a[id * cols + c];
    }
    out[id] = acc / float(cols);
  }
}

kernel void affon_reduce_axis_min_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    for (uint r = 1; r < rows; ++r) {
      best = min(best, a[r * cols + id]);
    }
    out[id] = best;
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    for (uint c = 1; c < cols; ++c) {
      best = min(best, a[id * cols + c]);
    }
    out[id] = best;
  }
}

kernel void affon_reduce_axis_min_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    long best = a[id];
    for (uint r = 1; r < rows; ++r) {
      best = min(best, a[r * cols + id]);
    }
    out[id] = best;
  } else {
    if (id >= rows) return;
    long best = a[id * cols];
    for (uint c = 1; c < cols; ++c) {
      best = min(best, a[id * cols + c]);
    }
    out[id] = best;
  }
}

kernel void affon_reduce_axis_max_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    for (uint r = 1; r < rows; ++r) {
      best = max(best, a[r * cols + id]);
    }
    out[id] = best;
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    for (uint c = 1; c < cols; ++c) {
      best = max(best, a[id * cols + c]);
    }
    out[id] = best;
  }
}

kernel void affon_reduce_axis_max_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    long best = a[id];
    for (uint r = 1; r < rows; ++r) {
      best = max(best, a[r * cols + id]);
    }
    out[id] = best;
  } else {
    if (id >= rows) return;
    long best = a[id * cols];
    for (uint c = 1; c < cols; ++c) {
      best = max(best, a[id * cols + c]);
    }
    out[id] = best;
  }
}

kernel void affon_reduce_axis_variance_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float mean = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      mean += a[r * cols + id];
    }
    mean /= float(rows);
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      const float d = a[r * cols + id] - mean;
      acc += d * d;
    }
    out[id] = acc / float(rows);
  } else {
    if (id >= rows) return;
    float mean = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      mean += a[id * cols + c];
    }
    mean /= float(cols);
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      const float d = a[id * cols + c] - mean;
      acc += d * d;
    }
    out[id] = acc / float(cols);
  }
}

kernel void affon_reduce_axis_std_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float mean = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      mean += a[r * cols + id];
    }
    mean /= float(rows);
    float acc = 0.0f;
    for (uint r = 0; r < rows; ++r) {
      const float d = a[r * cols + id] - mean;
      acc += d * d;
    }
    out[id] = sqrt(acc / float(rows));
  } else {
    if (id >= rows) return;
    float mean = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      mean += a[id * cols + c];
    }
    mean /= float(cols);
    float acc = 0.0f;
    for (uint c = 0; c < cols; ++c) {
      const float d = a[id * cols + c] - mean;
      acc += d * d;
    }
    out[id] = sqrt(acc / float(cols));
  }
}

kernel void affon_reduce_axis_argmin_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const float v = a[r * cols + id];
      if (v < best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = float(best_idx);
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const float v = a[id * cols + c];
      if (v < best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = float(best_idx);
  }
}

kernel void affon_reduce_axis_argmin_i64_f32(
    device const float* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const float v = a[r * cols + id];
      if (v < best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = ulong(best_idx);
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const float v = a[id * cols + c];
      if (v < best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = ulong(best_idx);
  }
}

kernel void affon_reduce_axis_argmin_i64_i64(
    device const long* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    long best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const long v = a[r * cols + id];
      if (v < best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = ulong(best_idx);
  } else {
    if (id >= rows) return;
    long best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const long v = a[id * cols + c];
      if (v < best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = ulong(best_idx);
  }
}

kernel void affon_reduce_axis_argmax_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const float v = a[r * cols + id];
      if (v > best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = float(best_idx);
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const float v = a[id * cols + c];
      if (v > best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = float(best_idx);
  }
}

kernel void affon_reduce_axis_argmax_i64_f32(
    device const float* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    float best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const float v = a[r * cols + id];
      if (v > best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = ulong(best_idx);
  } else {
    if (id >= rows) return;
    float best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const float v = a[id * cols + c];
      if (v > best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = ulong(best_idx);
  }
}

kernel void affon_reduce_axis_argmax_i64_i64(
    device const long* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  if (axis == 0) {
    if (id >= cols) return;
    long best = a[id];
    uint best_idx = 0;
    for (uint r = 1; r < rows; ++r) {
      const long v = a[r * cols + id];
      if (v > best) {
        best = v;
        best_idx = r;
      }
    }
    out[id] = ulong(best_idx);
  } else {
    if (id >= rows) return;
    long best = a[id * cols];
    uint best_idx = 0;
    for (uint c = 1; c < cols; ++c) {
      const long v = a[id * cols + c];
      if (v > best) {
        best = v;
        best_idx = c;
      }
    }
    out[id] = ulong(best_idx);
  }
}

kernel void affon_reduce_axis_nd_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* out_shape [[buffer(2)]],
    constant uint* out_strides [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& reduce_kind [[buffer(8)]], // 0=sum 1=mean 2=min 3=max 4=variance 5=std
    constant uint& out_len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= out_len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float acc = 0.0f;
  float best = 0.0f;
  if (reduce_kind == 2) {
    best = INFINITY;
  } else if (reduce_kind == 3) {
    best = -INFINITY;
  }

  float mean = 0.0f;
  if (reduce_kind == 4 || reduce_kind == 5) {
    for (uint j = 0; j < axis_size; ++j) {
      coord[axis] = j;
      uint off = 0;
      for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
      mean += a[off];
    }
    mean /= float(axis_size);
  }

  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const float v = a[off];
    if (reduce_kind == 0 || reduce_kind == 1) {
      acc += v;
    } else if (reduce_kind == 2) {
      best = min(best, v);
    } else if (reduce_kind == 3) {
      best = max(best, v);
    } else {
      const float dv = v - mean;
      acc += dv * dv;
    }
  }

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  if (reduce_kind == 0) out[out_off] = acc;
  else if (reduce_kind == 1) out[out_off] = acc / float(axis_size);
  else if (reduce_kind == 2) out[out_off] = best;
  else if (reduce_kind == 3) out[out_off] = best;
  else if (reduce_kind == 4) out[out_off] = acc / float(axis_size);
  else out[out_off] = sqrt(acc / float(axis_size));
}

kernel void affon_reduce_axis_nd_i64(
    device const long* a [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint* out_shape [[buffer(2)]],
    constant uint* out_strides [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& reduce_kind [[buffer(8)]], // 0=sum 2=min 3=max
    constant uint& out_len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= out_len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  long acc = 0;
  long best = 0;
  if (reduce_kind == 2) {
    best = LONG_MAX;
  } else if (reduce_kind == 3) {
    best = LONG_MIN;
  }

  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const long v = a[off];
    if (reduce_kind == 0) {
      acc += v;
    } else if (reduce_kind == 2) {
      best = min(best, v);
    } else {
      best = max(best, v);
    }
  }

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  if (reduce_kind == 0) out[out_off] = acc;
  else out[out_off] = best;
}

kernel void affon_reduce_axis_nd_arg_i64_f32(
    device const float* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint* out_shape [[buffer(2)]],
    constant uint* out_strides [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& choose_max [[buffer(8)]], // 0=min 1=max
    constant uint& out_len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= out_len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  coord[axis] = 0;
  uint off0 = 0;
  for (uint d = 0; d < ndim; ++d) off0 += coord[d] * input_strides[d];
  float best = a[off0];
  uint best_idx = 0;

  for (uint j = 1; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const float v = a[off];
    if ((choose_max != 0 && v > best) || (choose_max == 0 && v < best)) {
      best = v;
      best_idx = j;
    }
  }

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  out[out_off] = ulong(best_idx);
}

kernel void affon_reduce_axis_nd_arg_i64_i64(
    device const long* a [[buffer(0)]],
    device ulong* out [[buffer(1)]],
    constant uint* out_shape [[buffer(2)]],
    constant uint* out_strides [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& choose_max [[buffer(8)]], // 0 argmin, 1 argmax
    constant uint& out_len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= out_len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  coord[axis] = 0;
  uint off0 = 0;
  for (uint d = 0; d < ndim; ++d) off0 += coord[d] * input_strides[d];
  long best = a[off0];
  uint best_idx = 0;

  for (uint j = 1; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const long v = a[off];
    const bool better = choose_max != 0 ? (v > best) : (v < best);
    if (better) {
      best = v;
      best_idx = j;
    }
  }

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  out[out_off] = ulong(best_idx);
}

kernel void affon_matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  float acc = 0.0f;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * k + i] * b[i * n + gid.x];
  }
  out[gid.y * n + gid.x] = acc;
}

kernel void affon_matmul_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  long acc = 0;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * k + i] * b[i * n + gid.x];
  }
  out[gid.y * n + gid.x] = acc;
}

kernel void affon_matmul_strided_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& a_row_stride [[buffer(3)]],
    constant int& a_col_stride [[buffer(4)]],
    constant int& b_row_stride [[buffer(5)]],
    constant int& b_col_stride [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& k [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  float acc = 0.0f;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * a_row_stride + i * a_col_stride] * b[i * b_row_stride + gid.x * b_col_stride];
  }
  out[gid.y * n + gid.x] = acc;
}

kernel void affon_matmul_strided_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant int& a_row_stride [[buffer(3)]],
    constant int& a_col_stride [[buffer(4)]],
    constant int& b_row_stride [[buffer(5)]],
    constant int& b_col_stride [[buffer(6)]],
    constant uint& m [[buffer(7)]],
    constant uint& n [[buffer(8)]],
    constant uint& k [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  long acc = 0;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * a_row_stride + i * a_col_stride] * b[i * b_row_stride + gid.x * b_col_stride];
  }
  out[gid.y * n + gid.x] = acc;
}

kernel void affon_matmul_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  float acc = 0.0f;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * k + i] * b[i * n + gid.x];
  }
  const uint out_idx = gid.y * n + gid.x;
  out[out_idx] = acc + bias[out_idx];
}

kernel void affon_matmul_add_i64(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device const long* bias [[buffer(2)]],
    device long* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  long acc = 0;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * k + i] * b[i * n + gid.x];
  }
  const uint out_idx = gid.y * n + gid.x;
  out[out_idx] = acc + bias[out_idx];
}

kernel void affon_matmul_add_gelu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
  if (gid.x >= n || gid.y >= m) return;
  float acc = 0.0f;
  for (uint i = 0; i < k; ++i) {
    acc += a[gid.y * k + i] * b[i * n + gid.x];
  }
  const uint out_idx = gid.y * n + gid.x;
  const float x = acc + bias[out_idx];
  if (x >= 10.0f) {
    out[out_idx] = x;
    return;
  }
  if (x <= -10.0f) {
    out[out_idx] = 0.0f;
    return;
  }
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  out[out_idx] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void affon_muladd_inplace_f32(
    device float* target [[buffer(0)]],
    device const float* addend [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
  target[gid] = scale * target[gid] + addend[gid];
}

kernel void affon_axpy_inplace_f32(
    device float* target [[buffer(0)]],
    device const float* addend [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
  target[gid] = target[gid] + scale * addend[gid];
}

kernel void affon_sub_inplace_f32(
    device float* target [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
  target[gid] = target[gid] - delta[gid];
}

kernel void affon_sub_inplace_i64(
    device long* target [[buffer(0)]],
    device const long* delta [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
  target[gid] = target[gid] - delta[gid];
}

kernel void affon_scale_inplace_f32(
    device float* target [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
  target[gid] = target[gid] * scale;
}

kernel void affon_attention_scores_f32(
    device const float* q [[buffer(0)]],
    device const float* k_t [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device const long* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant float& mask_fill_value [[buffer(5)]],
    constant uint& m [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    uint row [[thread_position_in_grid]]
) {
  if (row >= m) return;

  float max_v = -INFINITY;
  for (uint c = 0; c < n; ++c) {
    float dot = 0.0f;
    for (uint kk = 0; kk < k; ++kk) {
      dot += q[row * k + kk] * k_t[kk * n + c];
    }
    const uint idx = row * n + c;
    float v = dot * scale[idx];
    if (mask[idx] != 0) v = mask_fill_value;
    out[idx] = v;
    max_v = max(max_v, v);
  }

  float sum_exp = 0.0f;
  for (uint c = 0; c < n; ++c) {
    const uint idx = row * n + c;
    const float e = exp(out[idx] - max_v);
    out[idx] = e;
    sum_exp += e;
  }

  for (uint c = 0; c < n; ++c) {
    const uint idx = row * n + c;
    out[idx] = out[idx] / sum_exp;
  }
}

kernel void affon_unary_chain_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* op_codes [[buffer(2)]],
    constant float* clamp_mins [[buffer(3)]],
    constant float* clamp_maxs [[buffer(4)]],
    constant uint& stage_count [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  float v = input[id];
  for (uint i = 0; i < stage_count; ++i) {
    const uint op = op_codes[i];
    switch (op) {
      case 1: v = fabs(v); break;
      case 2: v = -v; break;
      case 3: v = max(v, 0.0f); break;
      case 4: v = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); break;
      case 5: v = clamp(v, clamp_mins[i], clamp_maxs[i]); break;
      case 6: v = exp(v); break;
      case 7: v = log(v); break;
      case 8: v = sqrt(v); break;
      case 9: v = v * (1.0f / (1.0f + exp(-v))); break;
      default: break;
    }
  }
  out[id] = v;
}

kernel void affon_unary_chain_i64(
    device const long* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint* op_codes [[buffer(2)]],
    constant long* clamp_mins [[buffer(3)]],
    constant long* clamp_maxs [[buffer(4)]],
    constant uint& stage_count [[buffer(5)]],
    constant uint& len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  long v = input[id];
  for (uint i = 0; i < stage_count; ++i) {
    const uint op = op_codes[i];
    switch (op) {
      case 1: v = v < 0 ? -v : v; break;
      case 2: v = -v; break;
      case 3: v = max(v, (long)0); break;
      case 4: v = v > 0 ? 1 : (v < 0 ? -1 : 0); break;
      case 5: v = clamp(v, clamp_mins[i], clamp_maxs[i]); break;
      default: break;
    }
  }
  out[id] = v;
}

kernel void affon_binary_then_unary_chain_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& binary_code [[buffer(3)]],
    constant uint* op_codes [[buffer(4)]],
    constant float* clamp_mins [[buffer(5)]],
    constant float* clamp_maxs [[buffer(6)]],
    constant uint& stage_count [[buffer(7)]],
    constant uint& len [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  float v = 0.0f;
  switch (binary_code) {
    case 1: v = lhs[id] + rhs[id]; break;
    case 2: v = lhs[id] - rhs[id]; break;
    case 3: v = lhs[id] * rhs[id]; break;
    case 4: v = lhs[id] / rhs[id]; break;
    default: v = lhs[id]; break;
  }
  for (uint i = 0; i < stage_count; ++i) {
    const uint op = op_codes[i];
    switch (op) {
      case 1: v = fabs(v); break;
      case 2: v = -v; break;
      case 3: v = max(v, 0.0f); break;
      case 4: v = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); break;
      case 5: v = clamp(v, clamp_mins[i], clamp_maxs[i]); break;
      case 6: v = exp(v); break;
      case 7: v = log(v); break;
      case 8: v = sqrt(v); break;
      case 9: v = v * (1.0f / (1.0f + exp(-v))); break;
      default: break;
    }
  }
  out[id] = v;
}

kernel void affon_binary_then_unary_chain_i64(
    device const long* lhs [[buffer(0)]],
    device const long* rhs [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint& binary_code [[buffer(3)]],
    constant uint* op_codes [[buffer(4)]],
    constant long* clamp_mins [[buffer(5)]],
    constant long* clamp_maxs [[buffer(6)]],
    constant uint& stage_count [[buffer(7)]],
    constant uint& len [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  long v = 0;
  switch (binary_code) {
    case 1: v = lhs[id] + rhs[id]; break;
    case 2: v = lhs[id] - rhs[id]; break;
    case 3: v = lhs[id] * rhs[id]; break;
    case 4: v = lhs[id] / rhs[id]; break;
    default: v = lhs[id]; break;
  }
  for (uint i = 0; i < stage_count; ++i) {
    const uint op = op_codes[i];
    switch (op) {
      case 1: v = v < 0 ? -v : v; break;
      case 2: v = -v; break;
      case 3: v = max(v, (long)0); break;
      case 4: v = v > 0 ? 1 : (v < 0 ? -1 : 0); break;
      case 5: v = clamp(v, clamp_mins[i], clamp_maxs[i]); break;
      default: break;
    }
  }
  out[id] = v;
}

kernel void affon_embedding_i64_f32(
    device const float* table [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& emb_dim [[buffer(4)]],
    constant uint& index_count [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= index_count * emb_dim) return;
  const uint row = id / emb_dim;
  const uint col = id % emb_dim;
  const long token = index[row];
  if (token < 0 || uint(token) >= vocab) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = table[uint(token) * emb_dim + col];
}

kernel void affon_embedding_i64_i64(
    device const long* table [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    constant uint& emb_dim [[buffer(4)]],
    constant uint& index_count [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= index_count * emb_dim) return;
  const uint row = id / emb_dim;
  const uint col = id % emb_dim;
  const long token = index[row];
  if (token < 0 || uint(token) >= vocab) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = table[uint(token) * emb_dim + col];
}

kernel void affon_cast_f32_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = input[id];
}

kernel void affon_cast_i64_i64(
    device const long* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = input[id];
}

kernel void affon_cast_f32_i64(
    device const float* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = long(trunc(input[id]));
}

kernel void affon_validate_cast_index_f32_i64(
    device const float* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  const float v = input[id];
  if (!isfinite(v)) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  const float rounded = rint(v);
  if (fabs(v - rounded) > 1e-6f) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = long(rounded);
}

kernel void affon_cast_i64_f32(
    device const long* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = float(input[id]);
}


kernel void affon_gather_f32(
    device const float* input [[buffer(0)]],
    device const float* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = 0;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    if (d == axis) {
      const int gather_idx = int(index[id]);
      if (gather_idx < 0 || uint(gather_idx) >= axis_size) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      coord = uint(gather_idx);
    }
    input_off += coord * input_strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_gather_i64_f32(
    device const float* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = 0;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    if (d == axis) {
      const long gather_idx = index[id];
      if (gather_idx < 0 || ulong(gather_idx) >= axis_size) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      coord = uint(gather_idx);
    }
    input_off += coord * input_strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_gather_i64_i64(
    device const long* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    constant uint& len [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = 0;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % shape[d];
    rem /= shape[d];
    if (d == axis) {
      const long gather_idx = index[id];
      if (gather_idx < 0 || ulong(gather_idx) >= axis_size) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      coord = uint(gather_idx);
    }
    input_off += coord * input_strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_index_select_axis0_f32(
    device const float* input [[buffer(0)]],
    device const float* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
  const uint len = rows * width;
  if (id >= len) return;
  const uint row = id / width;
  const uint col = id % width;
  const int gather_idx = int(index[row]);
  if (gather_idx < 0) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = input[uint(gather_idx) * width + col];
}

kernel void affon_index_select_axis0_i64_f32(
    device const float* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
  const uint len = rows * width;
  if (id >= len) return;
  const uint row = id / width;
  const uint col = id % width;
  const long gather_idx = index[row];
  if (gather_idx < 0) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = input[uint(gather_idx) * width + col];
}

kernel void affon_index_select_i64_f32(
    device const float* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& index_len [[buffer(7)]],
    constant uint& axis_size [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    constant uint& len [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = 0;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % out_shape[d];
    rem /= out_shape[d];
    if (d == axis) {
      if (coord >= index_len) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      const long gather_idx = index[coord];
      if (gather_idx < 0 || ulong(gather_idx) >= axis_size) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      coord = uint(gather_idx);
    }
    input_off += coord * input_strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_index_select_axis0_i64_i64(
    device const long* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
  const uint len = rows * width;
  if (id >= len) return;
  const uint row = id / width;
  const uint col = id % width;
  const long gather_idx = index[row];
  if (gather_idx < 0) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  out[id] = input[uint(gather_idx) * width + col];
}

kernel void affon_index_select_i64_i64(
    device const long* input [[buffer(0)]],
    device const long* index [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& index_len [[buffer(7)]],
    constant uint& axis_size [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    constant uint& len [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;
  uint rem = id;
  uint input_off = 0;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    uint coord = rem % out_shape[d];
    rem /= out_shape[d];
    if (d == axis) {
      if (coord >= index_len) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      const long gather_idx = index[coord];
      if (gather_idx < 0 || ulong(gather_idx) >= axis_size) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
        return;
      }
      coord = uint(gather_idx);
    }
    input_off += coord * input_strides[d];
  }
  out[id] = input[input_off];
}

kernel void affon_index_select_axis0_scatter_add_f32(
    device const float* index [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& dst_rows [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  const uint len = rows * width;
  if (id >= len) return;
  const uint row = id / width;
  const uint col = id % width;
  const int scatter_idx = int(index[row]);
  if (scatter_idx < 0 || uint(scatter_idx) >= dst_rows) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    return;
  }
  atomic_fetch_add_explicit(&out[uint(scatter_idx) * width + col], src[id], memory_order_relaxed);
}

kernel void affon_topk_f32(
    device const float* input [[buffer(0)]],
    device float* values [[buffer(1)]],
    device float* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (k == 0 || k > max_k) return;

  float best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = -INFINITY;
    best_idx[i] = 0;
  }

  if (axis == 0) {
    if (id >= cols) return;
    for (uint r = 0; r < rows; ++r) {
      const float v = input[r * cols + id];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if (v > best_vals[i]) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = r;
      }
    }
    for (uint i = 0; i < k; ++i) {
      const uint out_idx = i * cols + id;
      values[out_idx] = best_vals[i];
      indices[out_idx] = float(best_idx[i]);
    }
  } else {
    if (id >= rows) return;
    const uint base = id * cols;
    for (uint c = 0; c < cols; ++c) {
      const float v = input[base + c];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if (v > best_vals[i]) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = c;
      }
    }
    const uint out_base = id * k;
    for (uint i = 0; i < k; ++i) {
      values[out_base + i] = best_vals[i];
      indices[out_base + i] = float(best_idx[i]);
    }
  }
}

kernel void affon_topk_i64_f32(
    device const float* input [[buffer(0)]],
    device float* values [[buffer(1)]],
    device ulong* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (k == 0 || k > max_k) return;

  float best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = -INFINITY;
    best_idx[i] = 0;
  }

  if (axis == 0) {
    if (id >= cols) return;
    for (uint r = 0; r < rows; ++r) {
      const float v = input[r * cols + id];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if (v > best_vals[i]) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = r;
      }
    }
    for (uint i = 0; i < k; ++i) {
      const uint out_idx = i * cols + id;
      values[out_idx] = best_vals[i];
      indices[out_idx] = ulong(best_idx[i]);
    }
  } else {
    if (id >= rows) return;
    const uint base = id * cols;
    for (uint c = 0; c < cols; ++c) {
      const float v = input[base + c];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if (v > best_vals[i]) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = c;
      }
    }
    for (uint i = 0; i < k; ++i) {
      const uint out_idx = id * k + i;
      values[out_idx] = best_vals[i];
      indices[out_idx] = ulong(best_idx[i]);
    }
  }
}

kernel void affon_topk_dir_i64_f32(
    device const float* input [[buffer(0)]],
    device float* values [[buffer(1)]],
    device ulong* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& largest [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (k == 0 || k > max_k) return;

  float best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = largest != 0 ? -INFINITY : INFINITY;
    best_idx[i] = 0;
  }

  if (axis == 0) {
    if (id >= cols) return;
    for (uint r = 0; r < rows; ++r) {
      const float v = input[r * cols + id];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if ((largest != 0 && v > best_vals[i]) || (largest == 0 && v < best_vals[i])) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = r;
      }
    }
    for (uint i = 0; i < k; ++i) {
      const uint out_idx = i * cols + id;
      values[out_idx] = best_vals[i];
      indices[out_idx] = ulong(best_idx[i]);
    }
  } else {
    if (id >= rows) return;
    const uint base = id * cols;
    for (uint c = 0; c < cols; ++c) {
      const float v = input[base + c];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        if ((largest != 0 && v > best_vals[i]) || (largest == 0 && v < best_vals[i])) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = c;
      }
    }
    const uint out_base = id * k;
    for (uint i = 0; i < k; ++i) {
      values[out_base + i] = best_vals[i];
      indices[out_base + i] = ulong(best_idx[i]);
    }
  }
}

kernel void affon_topk_nd_i64_f32(
    device const float* input [[buffer(0)]],
    device float* values [[buffer(1)]],
    device ulong* indices [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint* out_strides [[buffer(4)]],
    constant uint* input_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& axis [[buffer(7)]],
    constant uint& axis_size [[buffer(8)]],
    constant uint& k [[buffer(9)]],
    constant uint& largest [[buffer(10)]],
    constant uint& out_len [[buffer(11)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (id >= out_len) return;
  if (k == 0 || k > max_k) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = largest != 0 ? -INFINITY : INFINITY;
    best_idx[i] = 0;
  }

  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const float v = input[off];
    uint insert = k;
    for (uint i = 0; i < k; ++i) {
      if ((largest != 0 && v > best_vals[i]) || (largest == 0 && v < best_vals[i])) {
        insert = i;
        break;
      }
    }
    if (insert < k) {
      for (uint t = k - 1; t > insert; --t) {
        best_vals[t] = best_vals[t - 1];
        best_idx[t] = best_idx[t - 1];
      }
      best_vals[insert] = v;
      best_idx[insert] = j;
    }
  }

  coord[axis] = out_axis_coord;
  uint out_off = 0;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  values[out_off] = best_vals[out_axis_coord];
  indices[out_off] = ulong(best_idx[out_axis_coord]);
}

kernel void affon_topk_dir_i64_i64(
    device const long* input [[buffer(0)]],
    device long* values [[buffer(1)]],
    device long* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& largest [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (k == 0 || k > max_k) return;

  long best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = largest == 1 ? LONG_MIN : LONG_MAX;
    best_idx[i] = 0;
  }

  if (axis == 0) {
    if (id >= cols) return;
    for (uint r = 0; r < rows; ++r) {
      const long v = input[r * cols + id];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        const bool better = (largest == 1) ? (v > best_vals[i]) : (v < best_vals[i]);
        if (better) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = r;
      }
    }
    for (uint i = 0; i < k; ++i) {
      const uint out_idx = i * cols + id;
      values[out_idx] = best_vals[i];
      indices[out_idx] = long(best_idx[i]);
    }
  } else {
    if (id >= rows) return;
    const uint base = id * cols;
    for (uint c = 0; c < cols; ++c) {
      const long v = input[base + c];
      uint insert = k;
      for (uint i = 0; i < k; ++i) {
        const bool better = (largest == 1) ? (v > best_vals[i]) : (v < best_vals[i]);
        if (better) {
          insert = i;
          break;
        }
      }
      if (insert < k) {
        for (uint j = k - 1; j > insert; --j) {
          best_vals[j] = best_vals[j - 1];
          best_idx[j] = best_idx[j - 1];
        }
        best_vals[insert] = v;
        best_idx[insert] = c;
      }
    }
    const uint out_base = id * k;
    for (uint i = 0; i < k; ++i) {
      values[out_base + i] = best_vals[i];
      indices[out_base + i] = long(best_idx[i]);
    }
  }
}

kernel void affon_topk_nd_i64_i64(
    device const long* input [[buffer(0)]],
    device long* values [[buffer(1)]],
    device long* indices [[buffer(2)]],
    constant uint* out_shape [[buffer(3)]],
    constant uint* out_strides [[buffer(4)]],
    constant uint* input_strides [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant uint& axis [[buffer(7)]],
    constant uint& axis_size [[buffer(8)]],
    constant uint& k [[buffer(9)]],
    constant uint& largest [[buffer(10)]],
    constant uint& out_len [[buffer(11)]],
    uint id [[thread_position_in_grid]]
) {
  constexpr uint max_k = 64;
  if (id >= out_len) return;
  if (k == 0 || k > max_k) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }
  const uint out_axis_coord = coord[axis];

  long best_vals[max_k];
  uint best_idx[max_k];
  for (uint i = 0; i < k; ++i) {
    best_vals[i] = largest == 1 ? LONG_MIN : LONG_MAX;
    best_idx[i] = 0;
  }

  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const long v = input[off];
    uint insert = k;
    for (uint i = 0; i < k; ++i) {
      const bool better = (largest == 1) ? (v > best_vals[i]) : (v < best_vals[i]);
      if (better) {
        insert = i;
        break;
      }
    }
    if (insert < k) {
      for (uint r = k - 1; r > insert; --r) {
        best_vals[r] = best_vals[r - 1];
        best_idx[r] = best_idx[r - 1];
      }
      best_vals[insert] = v;
      best_idx[insert] = j;
    }
  }

  coord[axis] = out_axis_coord;
  uint out_off = 0;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * out_strides[d];
  values[out_off] = best_vals[out_axis_coord];
  indices[out_off] = long(best_idx[out_axis_coord]);
}

kernel void affon_one_hot_i64_f32(
    device const long* index [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_indices [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  const uint out_len = num_indices * num_classes;
  if (id >= out_len) return;
  const uint row = id / num_classes;
  const uint cls = id % num_classes;
  const long idx = index[row];
  if (idx < 0 || ulong(idx) >= num_classes) {
    atomic_store_explicit(status, 1u, memory_order_relaxed);
    out[id] = 0.0f;
    return;
  }
  out[id] = (uint(idx) == cls) ? 1.0f : 0.0f;
}

kernel void affon_reduce_sum_parallel_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  float acc = 0.0f;
  for (uint i = lid; i < len; i += tgs) { acc += a[i]; }
  shmem[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[0] = shmem[0];
}

kernel void affon_reduce_mean_parallel_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  float acc = 0.0f;
  for (uint i = lid; i < len; i += tgs) { acc += a[i]; }
  shmem[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[0] = shmem[0] / float(len);
}

kernel void affon_reduce_axis_sum_parallel_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  float acc = 0.0f;
  if (axis == 0) {
    for (uint r = lid; r < rows; r += tgs) { acc += a[r * cols + tgid]; }
  } else {
    for (uint c = lid; c < cols; c += tgs) { acc += a[tgid * cols + c]; }
  }
  shmem[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[tgid] = shmem[0];
}

kernel void affon_reduce_axis_mean_parallel_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    threadgroup float* shmem [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
  float acc = 0.0f;
  if (axis == 0) {
    for (uint r = lid; r < rows; r += tgs) { acc += a[r * cols + tgid]; }
  } else {
    for (uint c = lid; c < cols; c += tgs) { acc += a[tgid * cols + c]; }
  }
  shmem[lid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
    if (lid < stride) shmem[lid] += shmem[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) out[tgid] = (axis == 0) ? shmem[0] / float(rows) : shmem[0] / float(cols);
}


kernel void affon_layer_norm_nd_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* input_strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& axis_size [[buffer(6)]],
    constant uint& len [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % shape[d];
    rem /= shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float mean = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    mean += a[off];
  }
  mean /= float(axis_size);

  float var = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    float dv = a[off] - mean;
    var += dv * dv;
  }
  var /= float(axis_size);
  const float inv_std = rsqrt(var + eps);

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * input_strides[d];
  out[out_off] = (a[out_off] - mean) * inv_std;
}

kernel void affon_rms_norm_nd_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint* shape [[buffer(2)]],
    constant uint* input_strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    constant uint& axis [[buffer(5)]],
    constant uint& axis_size [[buffer(6)]],
    constant uint& len [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % shape[d];
    rem /= shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float mean_sq = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    const float x = a[off];
    mean_sq += x * x;
  }
  mean_sq /= float(axis_size);
  const float inv_rms = rsqrt(mean_sq + eps);

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * input_strides[d];
  out[out_off] = a[out_off] * inv_rms;
}


kernel void affon_add_layer_norm_nd_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    constant uint* input_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant uint& axis [[buffer(6)]],
    constant uint& axis_size [[buffer(7)]],
    constant uint& len [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= len) return;

  uint coord[8];
  uint rem = id;
  for (uint rev = 0; rev < ndim; ++rev) {
    uint d = ndim - 1 - rev;
    coord[d] = rem % shape[d];
    rem /= shape[d];
  }
  const uint out_axis_coord = coord[axis];

  float mean = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    mean += a[off] + b[off];
  }
  mean /= float(axis_size);

  float var = 0.0f;
  for (uint j = 0; j < axis_size; ++j) {
    coord[axis] = j;
    uint off = 0;
    for (uint d = 0; d < ndim; ++d) off += coord[d] * input_strides[d];
    float dv = (a[off] + b[off]) - mean;
    var += dv * dv;
  }
  var /= float(axis_size);
  const float inv_std = rsqrt(var + eps);

  uint out_off = 0;
  coord[axis] = out_axis_coord;
  for (uint d = 0; d < ndim; ++d) out_off += coord[d] * input_strides[d];
  out[out_off] = ((a[out_off] + b[out_off]) - mean) * inv_std;
}
