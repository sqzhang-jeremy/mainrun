"""
Interactive script to test GPT model predictions for Hacker News titles.
Loads a trained model checkpoint and generates title completions based on user input.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from mainrun.train import GPT, GPTConfig, BPETokenizer, train_tokenizer, get_titles, Hyperparameters

def load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained model and tokenizer from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    args = checkpoint.get('hyperparameters', Hyperparameters())

    # Recreate tokenizer
    print("Recreating tokenizer...")
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token))

    # Recreate model
    print("Recreating model...")
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=0.0,  # No dropout during inference
        mlp_ratio=args.mlp_ratio,
    )
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {tok.vocab_size:,}")
    print(f"  Block size: {args.block_size}")
    print()

    return model, tok, cfg, device

@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int, temperature: float, top_k: int, device: str):
    """Generate text continuation from a prompt."""
    model.eval()

    # Encode the prompt
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate tokens
    for _ in range(max_new_tokens):
        # Crop context if needed
        x_crop = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size:]

        # Forward pass
        logits, _ = model(x_crop)
        logits = logits[:, -1, :] / temperature

        # Optionally apply top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        x = torch.cat([x, next_id], dim=1)

        # Check for EOS token
        if tok.decode([next_id.item()]) == "<eos>":
            break

    # Decode and return
    generated_ids = x[0].tolist()
    return tok.decode(generated_ids)

def interactive_mode(model, tok, cfg, device):
    """Run interactive generation mode."""
    print("=" * 70)
    print("HACKER NEWS TITLE GENERATOR")
    print("=" * 70)
    print("Enter keywords or the beginning of a title, and the model will complete it.")
    print("Commands:")
    print("  /temp <value>    - Set temperature (default: 0.8)")
    print("  /topk <value>    - Set top-k sampling (default: 50, 0 = disabled)")
    print("  /tokens <value>  - Set max tokens to generate (default: 50)")
    print("  /quit or /exit   - Exit the program")
    print("=" * 70)
    print()

    # Generation parameters
    temperature = 0.8
    top_k = 50
    max_new_tokens = 50

    while True:
        try:
            prompt = input("\n📝 Enter prompt: ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()

                if cmd in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break
                elif cmd == "/temp" and len(parts) == 2:
                    try:
                        temperature = float(parts[1])
                        print(f"✓ Temperature set to {temperature}")
                    except ValueError:
                        print("❌ Invalid temperature value")
                elif cmd == "/topk" and len(parts) == 2:
                    try:
                        top_k = int(parts[1])
                        print(f"✓ Top-k set to {top_k}")
                    except ValueError:
                        print("❌ Invalid top-k value")
                elif cmd == "/tokens" and len(parts) == 2:
                    try:
                        max_new_tokens = int(parts[1])
                        print(f"✓ Max tokens set to {max_new_tokens}")
                    except ValueError:
                        print("❌ Invalid max tokens value")
                else:
                    print("❌ Unknown command or invalid syntax")
                continue

            # Generate completion
            print(f"\n🤖 Generating (temp={temperature}, top_k={top_k}, max_tokens={max_new_tokens})...")
            print("-" * 70)

            # Generate multiple samples
            num_samples = 3
            for i in range(num_samples):
                completion = generate(model, tok, prompt, max_new_tokens, temperature, top_k, device)
                # Clean up output
                completion = completion.replace("<eos>", "").strip()
                print(f"{i+1}. {completion}")

            print("-" * 70)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test GPT model for Hacker News title generation")
    parser.add_argument("--checkpoint", type=str, default="./model_checkpoint.pt",
                        help="Path to model checkpoint file (default: ./model_checkpoint.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, default: auto-detect)")
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        print("\nTo create a checkpoint:")
        print("  1. Train a model: task train")
        print("  2. The script will look for 'model_checkpoint.pt' by default")
        print("  3. Or specify a custom path: python test_model.py --checkpoint /path/to/checkpoint.pt")
        return

    # Determine device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, tok, cfg, device = load_model_and_tokenizer(args.checkpoint, device)

    # Start interactive mode
    interactive_mode(model, tok, cfg, device)

if __name__ == "__main__":
    main()
