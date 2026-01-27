"""
Training script with checkpoint saving enabled.
Saves the best model checkpoint based on validation loss.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from mainrun.train import (
    Hyperparameters, configure_logging, get_titles, train_tokenizer,
    BPETokenizer, GPTConfig, GPT, get_batch, iter_full_split
)

def save_checkpoint(model, optimizer, scheduler, hyperparameters, epoch, step, val_loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hyperparameters': hyperparameters,
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath} (val_loss: {val_loss:.6f})")

def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logger = configure_logging(args.log_file)

    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles+val_titles, args.vocab_size, eos_token=eos_token))
    train_text = eos_token.join(train_titles) + eos_token
    val_text = eos_token.join(val_titles) + eos_token
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)

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
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        mlp_ratio  = args.mlp_ratio,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)

    # Weight decay with parameter groups
    decay_params = set()
    no_decay_params = set()
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            decay_params.add(module.weight)
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
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

    # Warmup + Cosine Decay scheduler
    warmup_steps = int(0.05 * max_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

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
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        for _ in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            step += 1
            xb, yb = get_batch(train_ids, args.block_size, args.batch_size, device)
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
                      learning_rate=scheduler.get_last_lr()[0])

            if step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_result",
                          step=step,
                          val_loss=val_loss,
                          epoch=epoch)

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, opt, scheduler, args, epoch, step, val_loss,
                        checkpoint_dir / "best_model.pt"
                    )

                # Save latest checkpoint
                save_checkpoint(
                    model, opt, scheduler, args, epoch, step, val_loss,
                    checkpoint_dir / "latest_model.pt"
                )

    final_val_loss = evaluate()
    logger.log("training_complete",
              final_val_loss=final_val_loss,
              best_val_loss=best_val_loss,
              total_time=time.time()-t0)

    # Save final checkpoint
    save_checkpoint(
        model, opt, scheduler, args, args.epochs, step, final_val_loss,
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
