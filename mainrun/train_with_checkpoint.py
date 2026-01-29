"""
Training script with checkpoint saving enabled.
Saves the best model checkpoint (model + tokenizer + config) based on validation loss.

Usage:
  python train_with_checkpoint.py --config configs/exp-sparse-b64.yaml
"""

import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from train import (
    Hyperparameters, configure_logging, get_titles, train_tokenizer,
    BPETokenizer, GPTConfig, GPT, get_batch_random, get_batch_sequential,
    get_batch_titles, iter_full_split
)


def save_checkpoint(model, optimizer, scheduler, hyperparameters, tokenizer,
                    epoch, step, val_loss, filepath):
    """Save model checkpoint with tokenizer and full config."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hyperparameters': hyperparameters,
        'tokenizer_json': tokenizer.tk.to_str(),  # serialize tokenizer
        'model_config': {
            'vocab_size': model.cfg.vocab_size,
            'block_size': model.cfg.block_size,
            'n_layer': model.cfg.n_layer,
            'n_head': model.cfg.n_head,
            'd_model': model.cfg.d_model,
            'dropout': model.cfg.dropout,
            'mlp_ratio': model.cfg.mlp_ratio,
            'activation': model.cfg.activation,
            'residual_scaling': model.cfg.residual_scaling,
            'attention_type': model.cfg.attention_type,
            'attn_intermediate_dim': model.cfg.attn_intermediate_dim,
            'local_attn_ctx': model.cfg.local_attn_ctx,
        },
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} (val_loss: {val_loss:.6f})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp-sparse-b64.yaml',
                        help='Path to config YAML file')
    cli_args = parser.parse_args()

    config_path = Path(cli_args.config)
    if config_path.exists():
        args = Hyperparameters.from_yaml(str(config_path))
        print(f"Loaded config from {config_path}")
    else:
        args = Hyperparameters()
        print(f"Config not found at {config_path}, using defaults")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logger = configure_logging(args.log_file)

    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(
        train_titles + val_titles, args.vocab_size,
        normalizer_type=args.normalizer,
        pre_tokenizer_type=args.pre_tokenizer,
        min_frequency=args.min_frequency,
        domain_tokens=args.domain_tokens,
        eos_token=eos_token,
    ))

    # EOS handling
    if args.eos_handling == "tokenize_append":
        eos_id = tok.stoi[eos_token]
        def encode_titles_with_eos(titles, tok, eos_id):
            ids = []
            for title in titles:
                ids.extend(tok.encode(title))
                ids.append(eos_id)
            return ids
        train_ids = torch.tensor(encode_titles_with_eos(train_titles, tok, eos_id), dtype=torch.long)
        val_ids = torch.tensor(encode_titles_with_eos(val_titles, tok, eos_id), dtype=torch.long)
        train_text = eos_token.join(train_titles) + eos_token
        val_text = eos_token.join(val_titles) + eos_token
    else:
        train_text = eos_token.join(train_titles) + eos_token
        val_text = eos_token.join(val_titles) + eos_token
        train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
        val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

    # Per-title tokens for title-aware batching
    eos_id = tok.stoi[eos_token]
    pad_id = tok.stoi["<pad>"]
    train_title_tokens = [tok.encode(title) for title in train_titles]

    batches = len(train_ids) // (args.block_size * args.batch_size)
    max_steps = args.epochs * batches
    eval_interval = batches // args.evals_per_epoch
    logger.log("dataset_info",
               titles_count=len(train_titles),
               epochs=args.epochs,
               batches_per_epoch=batches,
               tokens_per_epoch=len(train_ids),
               vocab_size=tok.vocab_size)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        activation=args.activation,
        residual_scaling=args.residual_scaling,
        attention_type=args.attention_type,
        attn_intermediate_dim=args.attn_intermediate_dim,
        local_attn_ctx=args.local_attn_ctx,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    # Setup optimizer
    if args.optimizer == "adamw":
        decay_params = set()
        no_decay_params = set()
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                decay_params.add(module.weight)
            if isinstance(module, (nn.LayerNorm,)):
                if module.weight is not None:
                    no_decay_params.add(module.weight)
            for name, param in module.named_parameters(recurse=False):
                if name.endswith("bias"):
                    no_decay_params.add(param)

        for param in model.parameters():
            if param not in decay_params:
                no_decay_params.add(param)

        decay_group = [p for p in decay_params if p.requires_grad]
        no_decay_group = [p for p in no_decay_params if p.requires_grad and p not in decay_params]

        opt = torch.optim.AdamW(
            [
                {"params": decay_group, "weight_decay": args.weight_decay},
                {"params": no_decay_group, "weight_decay": 0.0},
            ],
            lr=args.lr
        )
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup scheduler
    if args.scheduler == "warmup_cosine":
        warmup_steps = int(args.warmup_fraction * max_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction='sum')
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    # Checkpoint tracking
    best_val_loss = float('inf')
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    step = 0
    ptr = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            if args.batch_mode == "title_aware":
                xb, yb = get_batch_titles(train_title_tokens, eos_id, args.block_size,
                                          args.batch_size, pad_id, device)
                _, loss = model(xb, yb, ignore_index=pad_id)
            elif args.batch_mode == "random":
                xb, yb = get_batch_random(train_ids, args.block_size, args.batch_size, device)
                _, loss = model(xb, yb)
            else:
                xb, yb, ptr = get_batch_sequential(train_ids, ptr, args.block_size, args.batch_size, device)
                _, loss = model(xb, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            elapsed = time.time() - t0
            logger.log("training_step",
                      step=step,
                      max_steps=max_steps,
                      loss=loss.item(),
                      elapsed_time=elapsed,
                      learning_rate=scheduler.get_last_lr()[0],
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_result",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, opt, scheduler, args, tok, epoch, step, val_loss,
                        checkpoint_dir / "best_model.pt"
                    )

                # Save latest checkpoint
                save_checkpoint(
                    model, opt, scheduler, args, tok, epoch, step, val_loss,
                    checkpoint_dir / "latest_model.pt"
                )

    final_val_loss = evaluate()
    logger.log("training_complete",
              final_val_loss=final_val_loss,
              best_val_loss=best_val_loss,
              total_time=time.time() - t0)

    # Save final checkpoint
    save_checkpoint(
        model, opt, scheduler, args, tok, args.epochs, step, final_val_loss,
        checkpoint_dir / "final_model.pt"
    )

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation loss: {final_val_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
