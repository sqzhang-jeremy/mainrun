import utils
import argparse
import math, random, time
from dataclasses import dataclass
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders, normalizers
from tqdm import tqdm
import structlog

@dataclass
class Hyperparameters:
    # Model architecture
    block_size: int = 64
    vocab_size: int = 16_000
    n_layer: int = 8
    n_head: int = 9
    d_model: int = 576
    dropout: float = 0
    mlp_ratio: float = 5.0
    activation: str = "gelu"  # gelu or swiglu
    residual_scaling: bool = True  # scale residual projections by 1/sqrt(2*n_layer)

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    evals_per_epoch: int = 3
    batch_mode: str = "random"  # random or sequential

    # Optimizer settings
    optimizer: str = "adamw"
    scheduler: str = "warmup_cosine"
    warmup_fraction: float = 0.05

    # Tokenizer settings
    normalizer: str = "none"           # "none" | "nfkc"
    pre_tokenizer: str = "bytelevel"   # "bytelevel" | "whitespace"
    eos_handling: str = "string_join"  # "string_join" | "tokenize_append"
    min_frequency: int = 1             # Minimum token frequency (1 = keep all)
    domain_tokens: bool = False        # Add "Show HN:", "Ask HN:", "Launch HN:"

    # Fixed (cannot change)
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"

    @classmethod
    def from_yaml(cls, path: str) -> "Hyperparameters":
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = open(log_file, 'w')
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()
            
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)
    
    return DualLogger(file_handler)

logger = None

def get_titles(num_titles: int, seed: int, val_frac: float) -> str:
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data").shuffle(seed=seed)
    titles = [row["title"].strip() for row in ds.take(num_titles)]
    n = int(num_titles * (1 - val_frac))
    return titles[:n], titles[n:]

def get_batch_random(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    """Random batch sampling - samples random starting positions each batch."""
    max_start = len(split_ids) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([split_ids[s : s + block_size] for s in starts]).to(device)
    y = torch.stack([split_ids[s + 1 : s + 1 + block_size] for s in starts]).to(device)
    return x, y

def get_batch_sequential(split_ids: torch.Tensor, ptr: int, block_size: int, batch_size: int, device: torch.device):
    """Sequential batch sampling - advances through data in order."""
    span = block_size * batch_size + 1
    if ptr + span >= len(split_ids):
        ptr = 0
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y, ptr + block_size * batch_size

def get_batch_titles(title_tokens: list[list[int]], eos_id: int, block_size: int,
                     batch_size: int, pad_id: int, device: torch.device):
    """Sample complete titles for training. Each sample is one title + EOS, padded to block_size."""
    indices = torch.randint(0, len(title_tokens), (batch_size,))
    batch_x = []
    batch_y = []
    for idx in indices:
        tokens = title_tokens[idx] + [eos_id]
        if len(tokens) > block_size + 1:
            tokens = tokens[:block_size + 1]
        else:
            tokens = tokens + [pad_id] * (block_size + 1 - len(tokens))
        batch_x.append(tokens[:-1])
        batch_y.append(tokens[1:])
    return (torch.tensor(batch_x, dtype=torch.long, device=device),
            torch.tensor(batch_y, dtype=torch.long, device=device))

def iter_full_split(split_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    span = block_size * batch_size + 1
    for ptr in range(0, len(split_ids) - span + 1, span):
        batch = split_ids[ptr: ptr + span]
        x = batch[:-1].view(batch_size, block_size).to(device)
        y = batch[1:].view(batch_size, block_size).to(device)
        yield x, y

def train_tokenizer(titles: list[str], vocab_size: int,
                    normalizer_type: str = "none",
                    pre_tokenizer_type: str = "bytelevel",
                    min_frequency: int = 1,
                    domain_tokens: bool = False,
                    unk_token: str = "<unk>", pad_token: str = "<pad>",
                    eos_token: str = "<eos>") -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))

    # Normalizer
    if normalizer_type == "nfkc":
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Replace(Regex(r"\s+"), " "),
            normalizers.Strip(),
        ])

    # Pre-tokenizer
    if pre_tokenizer_type == "whitespace":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:  # bytelevel (default)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()

    # Special tokens
    special_tokens = [pad_token, eos_token, unk_token]
    if domain_tokens:
        special_tokens.extend(["Show HN:", "Ask HN:", "Launch HN:"])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(titles, trainer)
    return tokenizer

class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self): return self.tk.get_vocab_size()

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    mlp_ratio: float
    activation: str = "gelu"  # gelu or swiglu
    residual_scaling: bool = True  # scale residual projections by 1/sqrt(2*n_layer)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class SwiGLU(nn.Module):
    """SwiGLU activation from Shazeer 2020: https://arxiv.org/abs/2002.05202
    Uses (Swish(xW₁) ⊗ xW₃)W₂ with 8/3 expansion ratio for parameter parity.
    Achieves 0.02-0.04 lower perplexity than GELU across model sizes.
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # Use 8/3 ratio (2/3 of mlp_ratio) to maintain parameter count with 3 weight matrices
        d_ff = int(cfg.mlp_ratio * cfg.d_model * 2 / 3)
        self.w1 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU: (Swish(xW₁) ⊗ xW₃)W₂ where Swish = SiLU
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden_dim = int(cfg.mlp_ratio * cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg) if cfg.activation == "swiglu" else MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.residual_scaling:
            with torch.no_grad():
                scale = 1.0 / math.sqrt(2 * cfg.n_layer)
                for block in self.blocks:
                    block.attn.proj.weight.mul_(scale)
                    # Handle both MLP (net[2]) and SwiGLU (w2) output projections
                    if hasattr(block.mlp, 'net'):
                        block.mlp.net[2].weight.mul_(scale)
                    elif hasattr(block.mlp, 'w2'):
                        block.mlp.w2.weight.mul_(scale)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, ignore_index: int = -100):
        B, T = idx.size()
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction='mean', ignore_index=ignore_index)
        return logits, loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/best-0128-1814.yaml',
                        help='Path to config YAML file')
    cli_args = parser.parse_args()

    # Load from config file
    config_path = Path(__file__).parent / cli_args.config
    if config_path.exists():
        args = Hyperparameters.from_yaml(str(config_path))
        print(f"Loaded config from {config_path}")
    else:
        args = Hyperparameters()
        print(f"Config not found at {config_path}, using defaults")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    global logger
    logger = configure_logging(args.log_file)
    
    hyperparams_dict = vars(args)
    logger.log("hyperparameters_configured", **hyperparams_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log("device_info", device=device)

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    
    eos_token = "<eos>"
    tok = BPETokenizer(train_tokenizer(
        train_titles+val_titles, args.vocab_size,
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
        train_text = eos_token.join(train_titles) + eos_token  # needed for len(val_text) in evaluate()
        val_text = eos_token.join(val_titles) + eos_token
    else:
        # Original: string join then tokenize
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
        vocab_size = tok.vocab_size,
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        d_model    = args.d_model,
        dropout    = args.dropout,
        mlp_ratio  = args.mlp_ratio,
        activation = args.activation,
        residual_scaling = args.residual_scaling,
    )
    model = GPT(cfg).to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("model_info", parameters_count=model_params)
    
    # Setup optimizer based on config
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        # Weight decay with parameter groups for AdamW
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
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Setup scheduler based on config
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    elif args.scheduler == "warmup_cosine":
        warmup_steps = int(args.warmup_fraction * max_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

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

    step = 0
    ptr = 0  # pointer for sequential batch mode
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
            else:  # sequential
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
                      prnt=False)

            if step == 1 or step % eval_interval == 0 or step == max_steps:
                val_loss = evaluate()
                logger.log("validation_step",
                          step=step,
                          max_steps=max_steps,
                          loss=val_loss,
                          elapsed_time=elapsed)

if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, 'file_handler'):
            logger.file_handler.close()
