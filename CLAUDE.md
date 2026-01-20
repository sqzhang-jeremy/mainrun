# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mainrun is an ML engineering assessment framework for evaluating LLM training and optimization skills. The challenge: minimize validation loss of a GPT-2 style model trained on Hacker News headlines within 7 fixed epochs. Baseline to beat: **1.754 validation loss**.

## Commands

```bash
task train      # Run training (auto-checkpoints first, downloads dataset if needed)
task submit     # Create checkpoint, zip repo, upload to evaluation system
task download   # Pre-download the Hacker News dataset
task checkpoint # Manually create a git checkpoint
```

## Rules & Constraints

**Cannot change:**
- Number of epochs (7)
- Random seed (1337)
- Dataset or validation fraction (10%)
- The `evaluate()` function in train.py

**Cannot use:**
- Pre-trained weights
- Data augmentation

**Can modify:**
- Model architecture, initialization, tokenization
- Hyperparameters, optimizer, scheduler
- Training loop (except evaluate())

## Architecture

```
mainrun/
├── train.py           # Main training script - all optimization work happens here
├── download_dataset.py
├── data/              # Cached dataset (auto-downloaded)
└── logs/              # Training logs (JSON format)
scripts/
├── checkpoint.mjs     # Auto-commit changes before train/submit
└── submit.mjs         # Zip and upload to evaluation
```

### Model Structure (train.py)

- **GPT class**: Token embeddings → positional embeddings → transformer blocks → layer norm → output head
- **Block**: LayerNorm → CausalSelfAttention → LayerNorm → MLP (pre-norm architecture)
- **BPE tokenizer**: Trained on dataset with 16K vocab, uses `<eos>` to separate titles

### Default Hyperparameters

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| block_size | 128 | n_layer | 6 |
| batch_size | 64 | n_head | 8 |
| vocab_size | 16,000 | d_model | 512 |
| dropout | 0.1 | lr | 6e-3 |
| optimizer | SGD | scheduler | CosineAnnealingLR |

## Submission

Place a `report.pdf` in the `mainrun/` folder before running `task submit`. The report should document changes made, reasoning, and effects on training curves.
