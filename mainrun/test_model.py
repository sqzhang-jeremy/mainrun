"""
Interactive script to test a trained GPT model on Hacker News title generation.
Loads model + tokenizer from a checkpoint saved by train_with_checkpoint.py.

Usage:
  python test_model.py --checkpoint ./checkpoints/best_model.pt
"""

import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPTConfig, GPT, BPETokenizer


def load_checkpoint(checkpoint_path: str, device: str = None):
    """Load model and tokenizer from a checkpoint file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore tokenizer from serialized JSON
    tok_json = checkpoint['tokenizer_json']
    tok = BPETokenizer(Tokenizer.from_str(tok_json))

    # Restore model config
    mc = checkpoint['model_config']
    cfg = GPTConfig(
        vocab_size=mc['vocab_size'],
        block_size=mc['block_size'],
        n_layer=mc['n_layer'],
        n_head=mc['n_head'],
        d_model=mc['d_model'],
        dropout=0.0,  # no dropout at inference
        mlp_ratio=mc['mlp_ratio'],
        activation=mc.get('activation', 'gelu'),
        residual_scaling=mc.get('residual_scaling', True),
        attention_type=mc.get('attention_type', 'causal'),
        attn_intermediate_dim=mc.get('attn_intermediate_dim', 0),
        local_attn_ctx=mc.get('local_attn_ctx', 16),
    )

    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_loss = checkpoint.get('val_loss', None)
    epoch = checkpoint.get('epoch', None)

    print(f"Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {tok.vocab_size:,}")
    print(f"  Block size: {cfg.block_size}")
    print(f"  Attention:  {cfg.attention_type}")
    if val_loss is not None:
        print(f"  Val loss:   {val_loss:.6f} (epoch {epoch})")
    print()

    return model, tok, cfg, device


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 60,
             temperature: float = 0.8, top_k: int = 50,
             device: str = "cpu"):
    """Generate text continuation from a prompt.

    Args:
        model: Trained GPT model (eval mode).
        tok: BPETokenizer instance.
        prompt: Input text to continue from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Keep only top-k logits before sampling (0 = no filtering).
        device: torch device string.

    Returns:
        The full generated string (prompt + continuation), with <eos> stripped.
    """
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to block_size if sequence is too long
        x_crop = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size:]

        logits, _ = model(x_crop)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        # Stop on <eos>
        token_str = tok.tk.decode([next_id.item()], skip_special_tokens=False)
        if "<eos>" in token_str:
            break

    generated_ids = x[0].tolist()
    return tok.decode(generated_ids)


def interactive_mode(model, tok, cfg, device):
    """Interactive REPL for generating Hacker News titles."""
    print("=" * 70)
    print("HACKER NEWS TITLE GENERATOR")
    print("=" * 70)
    print("Enter keywords or the beginning of a title to complete it.")
    print()
    print("Commands:")
    print("  /temp <value>    - Set temperature (default: 0.8)")
    print("  /topk <value>    - Set top-k sampling (default: 50, 0=off)")
    print("  /tokens <value>  - Set max new tokens (default: 60)")
    print("  /n <value>       - Set number of samples (default: 5)")
    print("  /quit            - Exit")
    print("=" * 70)
    print()

    temperature = 0.8
    top_k = 50
    max_new_tokens = 60
    num_samples = 3

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if not prompt:
                continue

            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()
                if cmd in ("/quit", "/exit", "/q"):
                    break
                elif cmd == "/temp" and len(parts) == 2:
                    temperature = float(parts[1])
                    print(f"  temperature = {temperature}")
                elif cmd == "/topk" and len(parts) == 2:
                    top_k = int(parts[1])
                    print(f"  top_k = {top_k}")
                elif cmd == "/tokens" and len(parts) == 2:
                    max_new_tokens = int(parts[1])
                    print(f"  max_new_tokens = {max_new_tokens}")
                elif cmd == "/n" and len(parts) == 2:
                    num_samples = int(parts[1])
                    print(f"  num_samples = {num_samples}")
                else:
                    print("  Unknown command")
                continue

            print(f"\n  [temp={temperature}, top_k={top_k}, tokens={max_new_tokens}]")
            print("-" * 70)
            for i in range(num_samples):
                result = generate(model, tok, prompt, max_new_tokens,
                                  temperature, top_k, device)
                result = result.replace("<eos>", "").strip()
                print(f"  {i+1}. {result}")
            print("-" * 70)
            print()

        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as e:
            print(f"  Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test trained GPT model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print()
        print("Train first:")
        print("  python train_with_checkpoint.py --config configs/exp-sparse-b64.yaml")
        return

    model, tok, cfg, device = load_checkpoint(args.checkpoint, args.device)
    interactive_mode(model, tok, cfg, device)


if __name__ == "__main__":
    main()
