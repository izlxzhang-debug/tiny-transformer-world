# Week 8 Research Log: Tiny Transformer Construction

**Status:** Implementation complete; overfitting smoke test passed

**Last updated:** 8 August 2026

## Work completed

- Implemented one self-contained PyTorch training program in
  [`train_tiny_transformer.py`](../../train_tiny_transformer.py).
- Built a lowercase word-level vocabulary from training stories only.
- Added `<pad>`, `<unk>`, and `<cls>` tokens.
- Added learned token and positional embeddings.
- Implemented explicit query, key, and value projections and scaled
  dot-product multi-head self-attention.
- Added padding masks so artificial padding cannot influence attention.
- Added configurable one- or two-layer transformer blocks with layer
  normalization, residual connections, GELU feed-forward networks, and an
  output classifier.
- Matched the classifier to the actual task's five answers: Lammy, Anneena,
  Jade, Penguin, and Nobody.
- Added cross-entropy training with AdamW, gradient clipping, deterministic
  sampling, periodic checkpoints, and checkpoint resumption.

## Attention dimensions

The default configuration uses batch size 20, model dimension 64, two heads,
and head dimension 32. For a selected sequence length of 136:

\[
X:[20,136,64]
\]

For one attention head:

\[
W_Q,W_K,W_V:[64,32]
\]

\[
Q,K,V:[20,136,32]
\]

\[
QK^\top:[20,136,136]
\]

\[
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{32}}\right)V
:[20,136,32]
\]

The implementation calculates both heads together and concatenates their
outputs back into tensors with final dimension 64.

## Overfitting experiment

The smoke test deterministically selects 100 training stories, with five
examples from every queried-object/answer combination. There are four queried
objects and five answer labels, giving 20 balanced cells.

Using seed 42, the default 78,277-parameter model reached 99% accuracy on the
same 100-example training subset after 21 epochs, exceeding the required 98%
threshold. The saved checkpoint was then loaded successfully and training was
resumed.

This experiment establishes that the model, data loader, masking, loss,
optimizer, and checkpoint system work together. Because evaluation uses the
same examples as training, it does not measure generalization.

## Run

From the repository root:

```bash
source .venv/bin/activate
python3 train_tiny_transformer.py
```

Resume from the saved checkpoint with:

```bash
python3 train_tiny_transformer.py --resume
```

Generated checkpoint files remain outside version control.
