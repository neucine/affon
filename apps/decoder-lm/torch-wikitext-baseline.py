#!/usr/bin/env python3
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def load_token_rows(path: str) -> list[list[int]]:
    parsed = json.loads(Path(path).read_text())
    if isinstance(parsed, dict):
        return parsed["rows"]
    return parsed


def pack_token_windows(rows: list[list[int]], seq_len: int, stride: int, join_with_token_id: int | None) -> list[list[int]]:
    flat: list[int] = []
    for index, row in enumerate(rows):
        flat.extend(row)
        if join_with_token_id is not None and index < len(rows) - 1:
            flat.append(join_with_token_id)
    width = seq_len + 1
    return [flat[start:start + width] for start in range(0, len(flat) - width + 1, stride)]


class AffonPrng:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next(self) -> int:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state


def shuffled_order(count: int, prng: AffonPrng) -> list[int]:
    order = list(range(count))
    for index in range(count - 1, 0, -1):
        other = prng.next() % (index + 1)
        order[index], order[other] = order[other], order[index]
    return order


def kaiming_uniform_affon_(param: torch.Tensor) -> None:
    fan_in = param.shape[0]
    bound = math.sqrt(6.0 / fan_in)
    nn.init.uniform_(param, -bound, bound)


class AffonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        kaiming_uniform_affon_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.weight) + self.bias


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.q_proj = AffonLinear(d_model, d_model)
        self.k_proj = AffonLinear(d_model, d_model)
        self.v_proj = AffonLinear(d_model, d_model)
        self.out_proj = AffonLinear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.up = AffonLinear(d_model, hidden_dim)
        self.down = AffonLinear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        normed = self.attn_norm(x)
        q = self.q_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(normed).reshape(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = q.matmul(k.permute(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        weights = torch.softmax(scores.masked_fill(mask, -1e9), dim=3)
        context = weights.matmul(v).permute(0, 2, 1, 3).contiguous().reshape(batch, seq, self.d_model)
        x = x + self.out_proj(context)
        ff = self.down(F.gelu(self.up(self.ff_norm(x)), approximate="none"))
        return x + ff


class DecoderLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        kaiming_uniform_affon_(self.token_embedding.weight)
        kaiming_uniform_affon_(self.position_embedding.weight)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.token_embedding.weight.matmul(x.reshape(batch * seq, x.shape[-1]).T).T
        return logits.reshape(batch, seq, self.token_embedding.num_embeddings)


@dataclass
class WarmupCosine:
    start: float
    peak: float
    end: float
    warmup_steps: int
    total_steps: int

    def lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            ratio = min(max(step / self.warmup_steps, 0.0), 1.0)
            return self.start + (self.peak - self.start) * ratio
        decay_progress = min(max(step - self.warmup_steps, 0), self.total_steps - self.warmup_steps)
        decay_ratio = decay_progress / (self.total_steps - self.warmup_steps)
        weight = (1.0 + math.cos(math.pi * decay_ratio)) / 2.0
        return self.end + (self.peak - self.end) * weight


@torch.no_grad()
def evaluate(model: DecoderLM, windows: list[list[int]], batch_size: int, max_batches: int | None, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    batches = math.ceil(len(windows) / batch_size)
    if max_batches is not None:
        batches = min(batches, max_batches)
    for batch_index in range(batches):
        batch = windows[batch_index * batch_size:(batch_index + 1) * batch_size]
        token_ids = torch.tensor(batch, dtype=torch.long, device=device)
        logits = model(token_ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), token_ids[:, 1:].reshape(-1), reduction="mean")
        total += float(loss.detach().cpu())
        count += 1
    model.train()
    return total / count


def vocab_size_from_tokenizer(path: str) -> int:
    parsed = json.loads(Path(path).read_text())
    vocab = parsed["model"]["vocab"]
    return max(int(value) for value in vocab.values()) + 1


def main() -> None:
    seed = env_int("AFFON_TORCH_BASELINE_SEED", 123)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(env_int("AFFON_TORCH_NUM_THREADS", 1))
    requested_device = os.environ.get("AFFON_TORCH_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested_device)

    seq_len = env_int("AFFON_TORCH_SEQ_LEN", 256)
    stride = env_int("AFFON_TORCH_STRIDE", 128)
    batch_size = env_int("AFFON_TORCH_BATCH_SIZE", 4)
    val_batch_size = env_int("AFFON_TORCH_VAL_BATCH_SIZE", 8)
    epochs = env_int("AFFON_TORCH_EPOCHS", 5)
    max_train_batches = env_int("AFFON_TORCH_MAX_TRAIN_BATCHES", 1250)
    max_eval_batches = env_int("AFFON_TORCH_MAX_EVAL_BATCHES", 20)
    max_train_batches_opt = None if max_train_batches <= 0 else max_train_batches
    max_eval_batches_opt = None if max_eval_batches <= 0 else max_eval_batches
    prefix = os.environ.get("AFFON_TORCH_PREFIX", "apps/decoder-lm/artifacts/wikitext/torch-baseline")

    train_rows = load_token_rows(os.environ.get("AFFON_TORCH_TRAIN_CACHE", "apps/decoder-lm/artifacts/wikitext/train-clean-cache.json"))
    val_rows = load_token_rows(os.environ.get("AFFON_TORCH_VAL_CACHE", "apps/decoder-lm/artifacts/wikitext/validation-clean-cache.json"))
    vocab_size = vocab_size_from_tokenizer(os.environ.get("AFFON_TORCH_TOKENIZER", "apps/decoder-lm/data/wikitext-gpt2-tokenizer.json"))
    eos_id = 50256
    train_windows = pack_token_windows(train_rows, seq_len, stride, eos_id)
    val_windows = pack_token_windows(val_rows, seq_len, stride, eos_id)

    model = DecoderLM(vocab_size, 256, 6, 8, 1024, seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    schedule = WarmupCosine(1e-5, 1e-4, 2e-6, 500, 37888)
    prng = AffonPrng(env_int("AFFON_TORCH_SHUFFLE_SEED", 123))
    history = []
    step = 0

    print(f"device={device} train_windows={len(train_windows)} val_windows={len(val_windows)} vocab_size={vocab_size}")
    initial_val = evaluate(model, val_windows, val_batch_size, max_eval_batches_opt, device)
    print(f"initial_val_loss={initial_val:.6f}")

    for epoch in range(1, epochs + 1):
        order = shuffled_order(len(train_windows), prng)
        batches = math.ceil(len(order) / batch_size)
        if max_train_batches_opt is not None:
            batches = min(batches, max_train_batches_opt)
        epoch_loss = 0.0
        model.train()
        for batch_index in range(batches):
            batch_order = order[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch = [train_windows[index] for index in batch_order]
            token_ids = torch.tensor(batch, dtype=torch.long, device=device)
            lr = schedule.lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), token_ids[:, 1:].reshape(-1), reduction="mean")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            batch_loss = float(loss.detach().cpu())
            epoch_loss += batch_loss
            if batch_index == 0 or (batch_index + 1) == batches or (batch_index + 1) % env_int("AFFON_TORCH_LOG_EVERY", 25) == 0:
                print(f"epoch {epoch} batch {batch_index + 1}/{batches}: step={step} loss={batch_loss:.6f} lr={lr:.6e} grad_norm={float(grad_norm):.6f}", flush=True)

        train_loss = epoch_loss / batches
        val_loss = evaluate(model, val_windows, val_batch_size, max_eval_batches_opt, device)
        history.append({"epoch": epoch, "step": step, "trainLoss": train_loss, "valLoss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

    summary = {
        "device": str(device),
        "epochs": epochs,
        "steps": step,
        "initialValLoss": initial_val,
        "finalTrainLoss": history[-1]["trainLoss"],
        "finalValLoss": history[-1]["valLoss"],
        "history": history,
    }
    Path(f"{prefix}-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
