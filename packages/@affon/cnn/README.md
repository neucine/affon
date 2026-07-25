# @affon/cnn

CNN architecture package scaffold.

This package is reserved for reusable convolutional architecture pieces if CNN work becomes real.

Expected ownership:

- convolutional blocks
- pooling blocks
- residual blocks
- common CNN stack pieces

Not owned here:

- image-domain workflows, which belong in `../vision/` or `../../../apps/`
- runtime convolution kernels, which belong in `affon` only after package pressure proves they are substrate-level primitives
