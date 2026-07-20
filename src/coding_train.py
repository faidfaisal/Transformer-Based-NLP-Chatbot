#!/usr/bin/env python3
"""
Train a GPT-2 style language model on code stored in JSONL files.

- Works with JSONL created from Project CodeNet (or any JSONL with a `text` field).
- Supports two dataset modes:
  1) Regular samples (one JSONL line -> one training sample, padded/truncated to --max_len)
  2) Packed streaming (--packed 1): concatenates many files and yields fixed-length blocks with no pad waste
- Adds tqdm progress bars for train/eval with running loss + LR
- Safe checkpoint init (`--init_from`) that handles vocab-size mismatches by partially copying
  embedding and output heads, and uses safe `weights_only=True` when available.

python .\coding_train.py `
  --train .\out\train.jsonl `
  --val .\out\val.jsonl `
  --model gpt2 `
  --output_dir .\runs\codenet-gpt2 `
  --epochs 3 --batch_size 2 --grad_accum 8 --max_len 1024 `
  --lr 5e-5 --warmup_ratio 0.03 --weight_decay 0.01 --clip_norm 1.0 `
  --packed 1 --lang_tag 1 `
  --init_from .\chatbot_gpt2_best.pth
  """
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm

# ------------------------ ARGS ------------------------ #

def get_args():
    ap = argparse.ArgumentParser("Train GPT-2 on code JSONL")
    ap.add_argument("--train", type=str, required=True, help="Path to train.jsonl")
    ap.add_argument("--val", type=str, default=None, help="Optional path to val.jsonl")
    ap.add_argument("--model", type=str, default="gpt2", help="HF model id or local path")
    ap.add_argument("--output_dir", type=Path, required=True, help="Where to save checkpoints")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--clip_norm", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (set 0 on Windows if issues)")

    ap.add_argument("--packed", type=int, default=1, help="1 = packed streaming, 0 = regular samples")
    ap.add_argument("--lang_tag", type=int, default=1, help="1 = add <lang=...> to tokenizer & prefix")

    ap.add_argument("--init_from", type=str, default=None, help="Path to .pth state_dict to initialize weights from")

    ap.add_argument("--save_best", type=int, default=1, help="Save best.ckpt.pth when val improves")
    ap.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps to use with iterable loaders (e.g., --packed 1)")
    ap.add_argument("--est_steps", type=int, default=0,help="If >0, use as tqdm total for IterableDataset to show ETA")
    ap.add_argument("--use_desc", type=int, default=0, help="1 = prepend problem description to code during training")

    return ap.parse_args()

# ------------------------ UTIL ------------------------ #

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_id: str, lang_tag: bool) -> GPT2TokenizerFast:
    tok: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    specials = ["<desc>", "<code>"]  # NEW markers
    if lang_tag:
        specials += ["<lang=C>", "<lang=C++>"]
    if specials:
        tok.add_special_tokens({"additional_special_tokens": specials})
    return tok


def _safe_load_init_weights(model: GPT2LMHeadModel, init_path: Path):
    """Load a state_dict that may have different vocab size.
    Copies overlapping rows for transformer.wte.weight and lm_head.weight, then loads the rest with strict=False.
    Uses weights_only=True if available to avoid unsafe pickle code execution.
    """
    print(f"Loading initial weights from {init_path}")
    # Safe load if supported (PyTorch >= 2.4)
    try:
        state = torch.load(init_path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        state = torch.load(init_path, map_location="cpu")

    # Some checkpoints are wrapped
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model_sd = model.state_dict()

    # If shapes differ, partially copy rows
    for k in ("transformer.wte.weight", "lm_head.weight"):
        if k in state and k in model_sd:
            old_w = state[k]
            new_w = model_sd[k]
            if isinstance(old_w, torch.Tensor) and isinstance(new_w, torch.Tensor) and old_w.ndim == 2 and new_w.ndim == 2:
                if old_w.shape != new_w.shape:
                    n = min(old_w.shape[0], new_w.shape[0])
                    with torch.no_grad():
                        new_w[:n].copy_(old_w[:n])
                    # Remove these keys so load_state_dict doesn't error
                    del state[k]
                    print(f"[init_from] Partially copied {k}: old {tuple(old_w.shape)} -> new {tuple(new_w.shape)} (rows={n})")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[init_from] Missing keys: {len(missing)} (expected when vocab differs)")
    if unexpected:
        print(f"[init_from] Unexpected keys: {len(unexpected)}")

# ------------------------ DATA ------------------------ #

class JSONLCodeDataset(Dataset):
    """Eager JSONL dataset. Each line must contain a `text` field.
       Optional fields: `lang` (e.g., C / C++), `path` (for reference).
    """
    def __init__(self, path: Path, tokenizer: GPT2TokenizerFast, max_len: int, lang_tag: bool):
        self.examples: List[Tuple[str, Optional[str]]] = []
        self.tok = tokenizer
        self.max_len = max_len
        self.lang_tag = lang_tag
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = str(obj.get("text", "")).rstrip()
                if not text:
                    continue
                lang = obj.get("lang")
                self.examples.append((text, lang))
                self._descs: List[Optional[str]] = []
                with path.open("r", encoding="utf-8", errors="ignore") as f2:
                    for line in f2:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        txt = str(obj.get("text", "")).rstrip()
                        if not txt:
                            continue
                        self._descs.append(obj.get("description"))  # may be None
                # If there was any mismatch in counts, pad to length
                while len(self._descs) < len(self.examples):
                    self._descs.append(None)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        text, lang = self.examples[idx]

        # Try to pull the raw dict again so we can access 'description' (we only stored (text, lang) in examples)
        # To avoid re-reading the file, we’ll encode description-less by default and only enrich if user asked.
        final_text = text
        if self.lang_tag and lang:
            final_text = f"<lang={lang}>\n" + final_text

        if getattr(self, "_use_desc", None) is None:
            # stash flag once (Dataset has no args directly; infer via tokenizer hack)
            # We store a boolean on the instance from an attribute we’ll set during construction in make_loaders().
            self._use_desc = False

        if self._use_desc:
            # We saved only (text, lang) in self.examples, so we need a side-channel to get 'description'.
            # Easiest: keep a parallel array with descriptions during dataset construction.
            # We'll assume we've been given: self._descs (same length as examples) by make_loaders().
            desc = ""
            if hasattr(self, "_descs"):
                desc = self._descs[idx] or ""
            if desc:
                # Build structured prompt
                parts = []
                if self.lang_tag and lang:
                    parts.append(f"<lang={lang}>")
                parts.append("<desc>")
                parts.append(desc.strip())
                parts.append("")  # blank line between sections
                parts.append("<code>")
                parts.append(text.strip())
                final_text = "\n".join(parts)

        enc = self.tok(
            final_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return input_ids, attn, labels


class PackedCodeIterable(IterableDataset):
    """Streams a JSONL and emits fixed-length blocks of token ids (no padding waste)."""
    def __init__(self, path: Path, tokenizer: GPT2TokenizerFast, block_len: int, lang_tag: bool):
        super().__init__()
        self.path = path
        self.tok = tokenizer
        self.block_len = block_len
        self.lang_tag = lang_tag
        self.eos_id = self.tok.eos_token_id

    def _encode_text(self, text: str, lang: Optional[str], desc: Optional[str], use_desc: bool) -> List[int]:
    # Build structured prompt:
    # <lang=LANG>\n<desc>\nDESC\n\n<code>\nCODE
        if use_desc and desc:
            pieces = []
            if self.lang_tag and lang:
                pieces.append(f"<lang={lang}>")
            pieces.append("<desc>")
            pieces.append(desc.strip())
            pieces.append("")  # blank line
            pieces.append("<code>")
            pieces.append(text.strip())
            to_encode = "\n".join(pieces)
        else:
            to_encode = text
            if self.lang_tag and lang:
                to_encode = f"<lang={lang}>\n{to_encode}"

        ids = self.tok.encode(to_encode, add_special_tokens=False)
        ids.append(self.eos_id)
        return ids

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        buffer: List[int] = []
        use_desc = getattr(self, "_use_desc", False)
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = str(obj.get("text", "")).rstrip()
                    if not text:
                        continue
                    lang = obj.get("lang")
                    desc = obj.get("description")
                except Exception:
                    continue
                buffer.extend(self._encode_text(text, lang, desc, use_desc))
                while len(buffer) >= self.block_len:
                    chunk = buffer[: self.block_len]
                    buffer = buffer[self.block_len :]
                    ids = torch.tensor(chunk, dtype=torch.long)
                    attn = torch.ones_like(ids)
                    yield ids, attn, ids.clone()
        # drop remainder (no partial blocks)


def make_loaders(train_path: Path, val_path: Optional[Path], tok: GPT2TokenizerFast, args):
    if args.packed:
        train_ds = PackedCodeIterable(train_path, tok, args.max_len, bool(args.lang_tag))
        collate = lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            torch.stack([b[1] for b in batch], dim=0),
            torch.stack([b[2] for b in batch], dim=0),
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0 if os.name == "nt" else args.num_workers,
            pin_memory=True, collate_fn=collate,
        )
    else:
        train_ds = JSONLCodeDataset(train_path, tok, args.max_len, bool(args.lang_tag))
        train_ds._use_desc = bool(args.use_desc)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0 if os.name == "nt" else args.num_workers,
            pin_memory=True,
        )

    val_loader = None
    if val_path and val_path.exists():
        if args.packed:
            val_ds = PackedCodeIterable(val_path, tok, args.max_len, bool(args.lang_tag))
            collate_v = lambda batch: (
                torch.stack([b[0] for b in batch], dim=0),
                torch.stack([b[1] for b in batch], dim=0),
                torch.stack([b[2] for b in batch], dim=0),
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0 if os.name == "nt" else args.num_workers,
                pin_memory=True, collate_fn=collate_v,
            )
        else:
            val_ds = JSONLCodeDataset(val_path, tok, args.max_len, bool(args.lang_tag))
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=0 if os.name == "nt" else args.num_workers,
                pin_memory=True,
            )
    return train_loader, val_loader

# ------------------------ EVAL ------------------------ #

def evaluate(model: GPT2LMHeadModel, loader: DataLoader, device: torch.device, use_amp: bool) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="[eval]", ncols=100, leave=False)
        for input_ids, attn, labels in pbar:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
            total += float(loss.item())
            n += 1
            avg_loss = total / n
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
    avg = total / max(n, 1)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl

# ------------------------ MAIN ------------------------ #

def main():
    args = get_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    # Tokenizer & model
    tok = build_tokenizer(args.model, bool(args.lang_tag))
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tok))

    # Optional: load from previous .pth checkpoint safely
    if args.init_from:
        init_path = Path(args.init_from)
        if init_path.exists():
            _safe_load_init_weights(model, init_path)
        else:
            print(f"[init_from] Path not found: {init_path}")

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    model.to(device)

    # Data
    train_path = Path(args.train)
    val_path = Path(args.val) if args.val else None
    train_loader, val_loader = make_loaders(train_path, val_path, tok, args)


    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Always create scaler + paths
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else torch.amp.GradScaler('cpu', enabled=False)
    best_val = float("inf")
    best_path = args.output_dir / "chatbot_gpt2_best.pth"
    last_path = args.output_dir / "chatbot_gpt2_last.pth"

    # Scheduler selection
    is_iterable = isinstance(getattr(train_loader, "dataset", None), IterableDataset)
    if is_iterable:
        # IterableDataset: unknown total steps -> constant + explicit warmup
        warmup_steps = max(1, int(args.warmup_steps))
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        # Map-style dataset: we can compute total updates and use linear decay
        steps_per_epoch = max(1, math.ceil(len(train_loader)))
        total_updates = math.ceil(steps_per_epoch * args.epochs / max(1, args.grad_accum))
        warmup_steps = max(1, int(args.warmup_ratio * total_updates))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        def _maybe_len(dl):
            try:
                return len(dl)
            except TypeError:
                return None  # IterableDataset has no length

        steps_per_epoch = _maybe_len(train_loader)
        if steps_per_epoch is None and args.est_steps > 0:
            steps_per_epoch = args.est_steps
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} [train]",
            ncols=110,
            total=steps_per_epoch  # None = no ETA, avoids crash
        )
        for step, batch in enumerate(pbar, start=1):
            input_ids, attn, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = model(input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()

            if (step % args.grad_accum) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += float(loss.item()) * max(1, args.grad_accum)
            avg_loss = running / step
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]

            # update progress bar
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")
        # Save last after each epoch
        torch.save(model.state_dict(), last_path)

        # Eval
        if val_loader is not None:
            val_loss, val_ppl = evaluate(model, val_loader, device, use_amp)
            tqdm.write(f"Epoch {epoch} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
            if args.save_best and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
                tqdm.write(f"✅ New best saved to {best_path}")
        else:
            tqdm.write(f"Epoch {epoch} | train_loss={avg_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
