# Third-Party Notices

Affon is licensed under MPL-2.0. The files below contain or derive from
third-party datasets and tokenizer assets distributed under their own terms.
Those terms apply to the corresponding files independently of Affon's source
code license.

## WikiText-2

Files:

- `apps/decoder-lm/data/wikitext-2-train.txt`
  - SHA-256: `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`
- `apps/decoder-lm/data/wikitext-2-validation.txt`
  - SHA-256: `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8`

Source: [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext)

License: Creative Commons Attribution-ShareAlike. The upstream dataset card
currently identifies the dataset as `cc-by-sa-3.0`; consult the upstream card
and source distribution for the terms applying to the downloaded revision.

Citation:

> Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher.
> “Pointer Sentinel Mixture Models.” arXiv:1609.07843, 2016.

The files are the tokenized WikiText-2 train and validation splits and retain
the upstream `<unk>` preprocessing.

## TinyStories

Files:

- `apps/decoder-lm/data/tinystories-train.txt`
  - SHA-256: `71ebf9cdac148f5a19f816e9e21cc1c9285e91b6b5e5c0369c61f765a480332b`
- `apps/decoder-lm/data/tinystories-validation.txt`
  - SHA-256: `b9f60c13a88617dcbd2c9174fe80aecb720ad033ac5fd975f59f980f22b34b5e`

Source: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

License: Community Data License Agreement – Sharing, Version 1.0
(`CDLA-Sharing-1.0`).

Citation:

> Ronen Eldan and Yuanzhi Li. “TinyStories: How Small Can Language Models Be
> and Still Speak Coherent English?” arXiv:2305.07759, 2023.

The repository contains small excerpts rather than the complete upstream
dataset.

## GPT-2 Tokenizer

File:

- `apps/decoder-lm/data/wikitext-gpt2-tokenizer.json`
  - SHA-256: `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`

Source: [openai-community/gpt2 tokenizer.json](https://huggingface.co/openai-community/gpt2/blob/main/tokenizer.json)

The file is byte-for-byte identical to the upstream `tokenizer.json` at the
time this notice was prepared. GPT-2 was released by OpenAI under its
[Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE).

## California Housing

File:

- `examples/california_housing.csv`
  - SHA-256: `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e`

Source: the California Housing dataset distributed through scikit-learn and
originally obtained from the StatLib repository.

Reference:

> R. Kelley Pace and Ronald Barry. “Sparse Spatial Autoregressions.”
> Statistics & Probability Letters 33, no. 3 (1997): 291–297.

The dataset is derived from the 1990 United States census. Consult the
[scikit-learn dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
for its description and upstream references.

## Locally Authored Fixtures

The following are small Affon project fixtures, not redistributed upstream
datasets:

- `apps/decoder-lm/data/helloworld-train.txt`
- `apps/decoder-lm/data/helloworld-tokenizer.json`
- `apps/decoder-lm/data/tinystories-tokenizer.json`

The TinyStories tokenizer was generated locally for the bundled sample.
