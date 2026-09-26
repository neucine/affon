"""Regenerate synthetic ByteLevel fixtures using tokenizers==0.22.2 (no downloads)."""
import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

assert tokenizers.__version__ == "0.22.2", "Use the recorded oracle version"
alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
vocab = {token: i for i, token in enumerate(alphabet)}
# Include merges which must not cross pre-tokenization boundaries, and an
# unreachable vocabulary entry to detect the incorrect whole-word shortcut.
merges = [("h", "i"), ("hi", "!"), ("Ċ", "H"), ("Ġ", "Ġ"), ("'", "s"), ("1", "2")]
for left, right in merges:
    vocab.setdefault(left + right, len(vocab))
vocab["hello"] = len(vocab)
prompts = ["", "hi!", "hello", " café\nHello!", "  hi  ", "\t hi\r\nHello\n\n", " \t\n",
           "I'm we've he'll she'd can't it's", "HELLO'S", "hi123!", "12 123", "中文🙂 café", "e\u0301",
           "hi<|endoftext|>Hello", "<|endoftext|><|endoftext|>", "hi<extra>!", "hi<extra-long>!",
           "hi  <strip> \tHello", "tag tagged #tag tag_ _tag", "\u00a0hi\u2003Hello"]
fixtures = []
for prefix, use_regex in [(False, True), (True, True), (False, False)]:
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=prefix, use_regex=use_regex)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken("<|endoftext|>", special=True, normalized=False)])
    tokenizer.add_tokens([AddedToken("<extra>", normalized=False), AddedToken("<extra-long>", normalized=False),
                          AddedToken("<strip>", lstrip=True, rstrip=True, normalized=False),
                          AddedToken("tag", single_word=True, normalized=False)])
    fixtures.append({"spec": json.loads(tokenizer.to_str()), "cases": [
        {"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids,
         "decoded": tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False),
         "skip_special": tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=True)}
        for text in prompts]})
path = Path(__file__).with_name("hf-bytelevel-reference.json")
path.write_text(json.dumps({"oracle": f"tokenizers=={tokenizers.__version__}", "fixtures": fixtures},
                           ensure_ascii=False, indent=2) + "\n")
print(path)
