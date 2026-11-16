#!/usr/bin/env python3
"""Train a small ByteLevel BPE tokenizer from Affon's WikiText training rows."""

import argparse
import re
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def rows(path: Path):
    for line in path.open(encoding="utf-8"):
        text = line.strip()
        if not text:
            continue
        text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace("<unk>", "")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("apps/decoder-lm/data/wikitext-2-train.txt"))
    parser.add_argument("--output", type=Path, default=Path("apps/decoder-lm/data/wikitext-8k-tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.train_from_iterator(rows(args.input), trainer=trainer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    print(f"Saved {tokenizer.get_vocab_size()} tokens to {args.output}")


if __name__ == "__main__":
    main()
