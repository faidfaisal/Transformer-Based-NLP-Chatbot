# Library functions — importing everything we need
import os  # operating system utilities (paths, env vars)
import math  # math helpers like exp() for perplexity
import random  # Python’s built-in random number generator
import torch  # main PyTorch library
import torch.nn as nn  # neural network modules (layers, losses)
from pathlib import Path  # convenient, cross‑platform path handling
from typing import List, Tuple, Optional, Dict  # type hints to make code clearer
from dataclasses import dataclass  # easy containers for structured data
from torch.utils.data import Dataset, DataLoader  # dataset / batching utilities
from torch.optim import AdamW  # Adam optimizer with weight decay (good for NLP)
from sklearn.model_selection import train_test_split  # split data into train/val
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup  # HF GPT‑2 + LR scheduler
from tqdm import tqdm  # progress bars so we can see training status
import re
from pathlib import Path
import pandas as pd

# Basic Predefined Global Variables — training knobs and file names
# ------------------------ CONFIG ------------------------ #
SEED = 42  # fixed seed so our runs are repeatable (same "random" each time)
MAX_LEN = 1024  # maximum number of tokens per example after tokenization
BATCH_SIZE = 2  # how many samples per GPU/CPU step (adjust if you have more VRAM)
GRAD_ACCUM = 8  # accumulate gradients over this many steps (effective batch = BATCH_SIZE * GRAD_ACCUM)
EPOCHS = 8  # how many full passes over the training data
LR = 5e-5  # learning rate for the optimizer (how big each update is)
WEIGHT_DECAY = 0.01  # L2 regularization that helps prevent overfitting
WARMUP_RATIO = 0.03  # % of total steps used to gradually ramp LR from 0 -> LR
CLIP_NORM = 1.0  # cap gradient norm to avoid exploding gradients
SAVE_BEST_TO = "chatbot_gpt2_best.pth"  # path to save the best model (lowest val loss)
LAST_CKPT = "chatbot_gpt2_last.pth"  # path to save the latest model each epoch

# DailyDialog files (Windows may hide .txt extension) — base names for splits
DD_TRAIN = "dialogues_train"  # training split base name
DD_VAL   = "dialogues_validation"  # validation split base name
DD_TEST  = "dialogues_test"  # optional final holdout test split base name

# Optional conditioning (set True/False) — whether to use acts/emotions/topics
USE_ACT   = True  # include dialogue act control tokens
USE_EMO   = True  # include emotion control tokens
USE_TOPIC = True  # include topic control tokens

# Where your files live; change if needed — point to your dataset directory
DATA_DIR = Path(".")  # use current folder by default; change to Path("/mnt/data") if needed

# --- Topical-Chat config ---
TOPICAL_CHAT_PATH = Path(r"C:\Users\Zheng\OneDrive\Desktop\ChatbotProj\topical_chat.csv")
USE_TOPICAL_CHAT = True            # turn ON to include the csv if present
DATA_MODE = "tc_only"                  # "dd_only" | "tc_only" | "mix"
TOPICAL_CHAT_RATIO = 0.8           # when DATA_MODE="mix": % of Topical-Chat in training set

# ------------------------ SEEDING ------------------------ #
random.seed(SEED)  # fix Python RNG
torch.manual_seed(SEED)  # fix PyTorch CPU RNG
torch.cuda.manual_seed_all(SEED)  # fix PyTorch GPU RNG (all devices)
# This section ensures everything "random" is repeatable between runs

# ------------------------ DEVICE ------------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pick GPU if present, else CPU
use_cuda_amp = device.type == "cuda"  # we’ll use AMP (mixed precision) only on CUDA
torch.backends.cudnn.benchmark = True  # speedup for fixed tensor sizes (OK for NLP with fixed MAX_LEN)
print(f"Using {device} (AMP={use_cuda_amp})")  # log what device and AMP setting we’re using

# ------------------------ LABEL MAPS ------------------------ #
# DailyDialog conventional mappings; we also accept raw ids if your files contain them
ACT_MAP = {  # map numeric act IDs to readable names
    1: "inform", 2: "question", 3: "directive", 4: "commissive"
}
EMO_MAP = {  # map numeric emotion IDs to readable names
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"
}
# Topics are 0..9 in the original paper; we’ll just use the id (string)

# ------------------------ TOKENIZER ------------------------ #
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # load the GPT‑2 tokenizer vocab + rules
# Use EOS as pad to keep shapes; GPT‑2 expects no pad by default
tokenizer.pad_token = tokenizer.eos_token  # pad token is EOS so shapes line up
tokenizer.padding_side = "left"  # left‑pad is better for decoder‑only models with attention masks

# Prepare special control tokens (we add only those we might need)
specials = []  # start with an empty list of extra tokens
if USE_ACT:
    specials += [f"<act={name}>" for name in sorted(set(ACT_MAP.values()))]  # add act tokens like <act=question>
if USE_EMO:
    specials += [f"<emo={name}>" for name in sorted(set(EMO_MAP.values()))]  # add emo tokens like <emo=happiness>
if USE_TOPIC:
    specials += [f"<topic={i}>" for i in range(0, 20)]  # allow up to 20 topic IDs; unused ones are harmless

if specials:
    tokenizer.add_special_tokens({"additional_special_tokens": specials})  # extend tokenizer with our control tokens

# ------------------------ DATA ------------------------ #
@dataclass  # simple container for a single training pair
class Pair:
    src: str  # the input utterance (User)
    tgt: str  # the target response (Bot)
    act: Optional[str] = None    # optional dialog act label for the target
    emo: Optional[str] = None    # optional emotion label for the target
    topic: Optional[str] = None  # optional topic ID for the dialog


def _read_text_lines(path: Path) -> List[List[str]]:
    dialogs = []  # will hold a list of dialogs; each dialog is a list of utterances
    with open(path, encoding="utf-8", errors="ignore") as f:  # open the text file safely
        for line in f:  # go line by line; each line is a whole dialog
            utts = [u.strip() for u in line.strip().split("__eou__") if u.strip()]  # split by end‑of‑utterance marker
            if len(utts) >= 2:  # keep only dialogs with at least two turns
                dialogs.append(utts)  # store the sequence of utterances for this dialog
    return dialogs  # return list of dialogs


def _read_int_lists(path: Path) -> List[List[int]]:
    """
    Each line is space-separated ints aligned with utterances of the dialog.
    """
    items = []  # will hold lists of ints (per dialog)
    with open(path, encoding="utf-8", errors="ignore") as f:  # open the labels file
        for line in f:  # read each dialog’s labels
            tokens = line.strip().split()  # split on spaces
            if tokens:  # if the line isn’t empty
                try:
                    items.append([int(t) for t in tokens])  # convert each token to int
                except ValueError:
                    # malformed line -> push empty to force skip
                    items.append([])  # if bad data, append empty so we can skip later
            else:
                items.append([])  # empty line -> no labels
    return items  # return list of label sequences


def _read_topic_per_dialog(path: Path) -> List[Optional[int]]:
    topics: List[Optional[int]] = []  # one topic per dialog (or None)
    with open(path, encoding="utf-8", errors="ignore") as f:  # open topic file
        for line in f:  # one topic ID per line (per dialog)
            line = line.strip()  # trim whitespace
            if not line:  # if the line is empty
                topics.append(None)  # topic is unknown for this dialog
                continue  # move to next line
            try:
                topics.append(int(line))  # parse the topic as int
            except ValueError:
                topics.append(None)  # if it’s not an int, store None
    return topics  # return list of topics (ints or None)


def _safe_map(did: int, uidx: int,
              acts: Optional[List[List[int]]],
              emos: Optional[List[List[int]]]) -> Tuple[Optional[str], Optional[str]]:
    act_name = emo_name = None  # default if not found
    if acts and did < len(acts) and uidx < len(acts[did]):  # check bounds for acts
        act_id = acts[did][uidx]  # get the act id for this utterance
        act_name = ACT_MAP.get(act_id) if act_id in ACT_MAP else None  # map to name if valid
    if emos and did < len(emos) and uidx < len(emos[did]):  # check bounds for emotions
        emo_id = emos[did][uidx]  # get the emotion id
        emo_name = EMO_MAP.get(emo_id) if emo_id in EMO_MAP else None  # map to name if valid
    return act_name, emo_name  # return readable labels (or None)


def load_pairs(split_basename: str, data_dir: Path = DATA_DIR) -> List[Pair]:
    """
    Build (src -> tgt) pairs from the specified split ("dialogues_train", etc.).
    If *_act / *_emotion / dialogues_topic exist, attach labels for the TARGET utterance.
    """
    text_path = data_dir / (split_basename if split_basename.endswith(".txt") else f"{split_basename}.txt")  # resolve file path
    dialogs = _read_text_lines(text_path)  # read the raw dialogs from file

    acts = emos = None  # default if we don’t have label files
    topic_list = None  # default if we don’t have topic file

    if USE_ACT:  # if we’re using act labels
        act_path = data_dir / (split_basename.replace("dialogues", "dialogues_act") + ("" if split_basename.endswith(".txt") else ".txt"))  # act labels path
        if act_path.exists():  # only read if the file exists
            acts = _read_int_lists(act_path)  # load act IDs aligned to utterances

    if USE_EMO:  # if we’re using emotion labels
        emo_path = data_dir / (split_basename.replace("dialogues", "dialogues_emotion") + ("" if split_basename.endswith(".txt") else ".txt"))  # emotion labels path
        if emo_path.exists():  # only read if the file exists
            emos = _read_int_lists(emo_path)  # load emotion IDs aligned to utterances

    if USE_TOPIC:  # topics are global (often a single file)
        # Topic file is single (not split by train/val/test) in many distributions
        topic_path = data_dir / ("dialogues_topic.txt" if (data_dir / "dialogues_topic.txt").exists() else "dialogues_topic")  # try both names
        if topic_path.exists():  # only read if present
            topic_list = _read_topic_per_dialog(topic_path)  # load one topic per dialog

    pairs: List[Pair] = []  # will store all (src->tgt) training pairs
    for did, utts in enumerate(dialogs):  # iterate through each dialog by ID
        topic_tok = None  # default topic string for this dialog
        if USE_TOPIC and topic_list and did < len(topic_list) and topic_list[did] is not None:  # if topic available
            topic_tok = str(topic_list[did])  # convert topic int to string token
        for i in range(len(utts) - 1):  # for each adjacent pair of utterances
            src = utts[i]  # the user’s line
            tgt = utts[i + 1]  # the bot’s reply (next line)
            act_name, emo_name = _safe_map(did, i + 1, acts, emos)  # labels for TARGET utterance
            pairs.append(Pair(src=src, tgt=tgt, act=act_name, emo=emo_name, topic=topic_tok))  # save the training pair
    return pairs  # give back all pairs

def load_topical_chat_pairs(csv_path: Path) -> List[Pair]:
    """
    Load Topical-Chat CSV and convert to (src -> tgt) Pair list.
    Tries to be robust to column name variants.
    """
    if not csv_path.exists():
        print(f"[TopicalChat] File not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)

    # Heuristic column detection
    # conversation id
    for cid in ["conversation_id", "conv_id", "dialog_id", "dlg_id", "conversationId", "cid"]:
        if cid in df.columns:
            conv_col = cid
            break
    else:
        # If no conversation id, treat entire file as one long dialog
        conv_col = None

    # text / utterance
    for tx in ["text", "utterance", "message", "msg", "response"]:
        if tx in df.columns:
            text_col = tx
            break
    else:
        raise ValueError("Topical-Chat CSV: couldn't find a text column (tried text/utterance/message/msg/response).")

    # topic (optional)
    topic_col = "topic" if "topic" in df.columns else None

    pairs: List[Pair] = []

    if conv_col is None:
        # Single dialog: just walk the rows in order
        utts = df[text_col].astype(str).fillna("").tolist()
        topic_val = str(df[topic_col].iloc[0]) if topic_col else None
        for i in range(len(utts) - 1):
            src, tgt = utts[i].strip(), utts[i + 1].strip()
            if src and tgt:
                pairs.append(Pair(src=src, tgt=tgt, topic=topic_val))
        return pairs

    # Multiple dialogs grouped by conversation
    for conv_id, g in df.groupby(conv_col, sort=False):
        g = g.sort_index()
        utts = g[text_col].astype(str).fillna("").tolist()
        topic_val = None
        if topic_col and topic_col in g.columns:
            # take the most common topic in the group
            try:
                topic_val = str(g[topic_col].mode(dropna=True).iloc[0])
            except Exception:
                topic_val = None

        for i in range(len(utts) - 1):
            src, tgt = utts[i].strip(), utts[i + 1].strip()
            if src and tgt:
                pairs.append(Pair(src=src, tgt=tgt, topic=topic_val))

    print(f"[TopicalChat] Loaded {len(pairs)} pairs from {csv_path}")
    return pairs

def mix_pairs(pairs_a: List[Pair], pairs_b: List[Pair], ratio_a: float, seed: int = SEED) -> List[Pair]:
    """
    Return a list mixing pairs_a and pairs_b with target fraction ratio_a for A.
    """
    random.seed(seed)
    if not pairs_a: return pairs_b
    if not pairs_b: return pairs_a

    n_total = len(pairs_a) + len(pairs_b)
    n_a = int(ratio_a * n_total)
    n_b = n_total - n_a
    sample_a = random.sample(pairs_a, min(n_a, len(pairs_a)))
    sample_b = random.sample(pairs_b, min(n_b, len(pairs_b)))
    mixed = sample_a + sample_b
    random.shuffle(mixed)
    return mixed

# ------------------------ DATASET ------------------------ #
class ChatDataset(Dataset):
    """
    Builds sequences like:
      "User: <src>  EOS  Bot: [<cond tokens>] <tgt>  EOS"
    and masks the loss so only the Bot span contributes.
    """
    def __init__(self, pairs: List[Pair], max_len=MAX_LEN):  # dataset takes list of pairs and a max length
        self.max_len = max_len  # store max sequence length
        self.samples = []  # will hold tuples of (input_ids, attention_mask, labels)
        eos = tokenizer.eos_token  # grab EOS token string once

        for p in pairs:  # build one sample per pair
            cond_tokens = []  # control tokens we’ll attach to "Bot:" (act/emo/topic)
            if USE_ACT and p.act:
                cond_tokens.append(f"<act={p.act}>")  # add act token if present
            if USE_EMO and p.emo:
                cond_tokens.append(f"<emo={p.emo}>")  # add emotion token if present
            if USE_TOPIC and p.topic is not None:
                cond_tokens.append(f"<topic={p.topic}>")  # add topic token if present

            cond = (" " + " ".join(cond_tokens)).strip()  # single space + join tokens (or empty)
            # put control codes immediately after "Bot:" so model learns to honor them
            text = f"User: {p.src} {eos} Bot:{(' ' + cond) if cond else ''} {p.tgt} {eos}"  # final linearized training text

            enc = tokenizer(  # tokenize the full text to ids + mask
                text,
                truncation=True,  # cut off if too long
                padding="max_length",  # pad to max_len for efficient batching
                max_length=self.max_len,  # target sequence length
                return_tensors="pt"  # return PyTorch tensors
            )
            input_ids = enc["input_ids"].squeeze(0)  # shape: (seq_len,)
            attn_mask = enc["attention_mask"].squeeze(0)  # 1 for real tokens, 0 for padding

            bot_ids = tokenizer(" Bot:", add_special_tokens=False)["input_ids"]  # token pattern for " Bot:"

            start_idx = None  # will hold index where " Bot:" starts
            for i in range(0, len(input_ids) - len(bot_ids) + 1):  # scan through sequence to find the pattern
                if torch.equal(input_ids[i:i+len(bot_ids)], torch.tensor(bot_ids, dtype=input_ids.dtype)):  # pattern match
                    start_idx = i  # found the beginning of the Bot span
                    break  # stop scanning

            labels = input_ids.clone()  # start labels as a copy of inputs
            ignore_index = -100  # PyTorch cross entropy ignores positions set to -100
            if start_idx is None:  # if we couldn’t find the Bot span (very rare)
                labels[:] = ignore_index  # ignore the whole sequence for loss
            else:
                bot_start = start_idx  # mark where Bot tokens begin
                for i in range(0, bot_start):  # for everything before Bot:
                    labels[i] = ignore_index  # don’t compute loss (we only train on Bot part)
                for i in range(len(input_ids)):  # also ignore padded tokens
                    if attn_mask[i] == 0:
                        labels[i] = ignore_index  # no loss on padding

            self.samples.append((input_ids, attn_mask, labels))  # store one training example

    def __len__(self):  # required by PyTorch — how many samples we have
        return len(self.samples)  # number of (input,mask,label) triples

    def __getitem__(self, idx):  # required by PyTorch — fetch one sample by index
        return self.samples[idx]  # return the (input_ids, attn_mask, labels) tuple


# ------------------------ MODEL ------------------------ #
model = GPT2LMHeadModel.from_pretrained("gpt2") 
model.resize_token_embeddings(len(tokenizer))  # extend embedding matrix to include our new control tokens
if Path("chatbot_gpt2_best.pth").exists():
    state_dict = torch.load("chatbot_gpt2_best.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass
model.to(device)


# ------------------------ TRAIN / EVAL LOOPS ------------------------ #
def evaluate(model, loader):  # run evaluation over a dataloader and compute avg loss + ppl
    model.eval()  # set model to eval mode (turn off dropout, etc.)
    total_loss = 0.0  # accumulator for loss
    count = 0  # how many batches we’ve seen
    with torch.no_grad():  # no gradients during eval
        for input_ids, attn_mask, labels in loader:  # loop over validation batches
            input_ids = input_ids.to(device)  # move to device
            attn_mask = attn_mask.to(device)  # move to device
            labels = labels.to(device)  # move to device
            with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):  # mixed precision on GPU
                out = model(input_ids, attention_mask=attn_mask, labels=labels)  # forward pass
                loss = out.loss  # cross‑entropy loss returned by the model
            total_loss += loss.item()  # add this batch’s loss to total
            count += 1  # increment batch counter
    avg = total_loss / max(count, 1)  # average loss across batches
    ppl = math.exp(avg) if avg < 20 else float("inf")  # convert loss to perplexity (guard from overflow)
    return avg, ppl  # return both metrics


def make_loaders(train_pairs, val_pairs):  # create DataLoaders for train and val sets
    train_ds = ChatDataset(train_pairs)  # wrap train pairs in our dataset
    val_ds   = ChatDataset(val_pairs)  # wrap val pairs in our dataset
    train_loader = DataLoader(  # create the training DataLoader
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  # shuffle for training
        num_workers=4, pin_memory=True, persistent_workers=True  # speed options for dataloader
    )
    val_loader = DataLoader(  # create the validation DataLoader
        val_ds, batch_size=BATCH_SIZE, shuffle=False,  # no shuffle for validation
        num_workers=4, pin_memory=True, persistent_workers=True  # same speed options
    )
    return train_loader, val_loader  # return both loaders


def train():  # main training function
    # --- Load DailyDialog pairs ---
    dd_train_pairs = load_pairs(DD_TRAIN, DATA_DIR)
    dd_val_pairs   = load_pairs(DD_VAL,   DATA_DIR) if (DATA_DIR / f"{DD_VAL}.txt").exists() else None

    # --- Load Topical-Chat pairs if enabled ---
    tc_pairs = []
    if USE_TOPICAL_CHAT and TOPICAL_CHAT_PATH.exists():
        tc_pairs = load_topical_chat_pairs(TOPICAL_CHAT_PATH)
    print(f"[Data] DD_train={len(dd_train_pairs)}  DD_val={len(dd_val_pairs) if dd_val_pairs else 0}  TC={len(tc_pairs)}")
    # --- Decide data according to DATA_MODE ---
    if DATA_MODE == "tc_only" and tc_pairs:
        all_pairs = tc_pairs
    elif DATA_MODE == "mix" and tc_pairs:
        # Mix Topical-Chat (A) and DailyDialog (B)
        all_pairs = mix_pairs(tc_pairs, dd_train_pairs, ratio_a=TOPICAL_CHAT_RATIO, seed=SEED)
    else:
        # Default: DailyDialog only
        all_pairs = dd_train_pairs

    # --- Train/Val split (if no official val split for chosen data) ---
    # If we picked tc_only or mix, there is no official val split -> split here
    if (DATA_MODE in ["tc_only", "mix"] and tc_pairs) or not dd_val_pairs:
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.05, random_state=SEED)
    else:
        # Using DailyDialog official split
        train_pairs, val_pairs = dd_train_pairs, dd_val_pairs

    train_loader, val_loader = make_loaders(train_pairs, val_pairs)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # set up AdamW optimizer
    t_total = (len(train_loader) * EPOCHS) // GRAD_ACCUM  # total number of optimizer steps we will take
    warmup_steps = max(1, int(WARMUP_RATIO * t_total))  # compute how many steps to warm up
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)  # LR schedule: warmup then linear decay

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)  # gradient scaler for mixed precision

    try:
        with open("best_loss.txt", "r") as f:
            best_val = float(f.read().strip())
        print(f"Loaded previous best_val: {best_val:.4f}")
    except FileNotFoundError:
        best_val = float("inf")
        print("No previous best_val found, starting fresh.")  # keep track of best (lowest) validation loss
    global_step = 0  # count optimizer steps (after grad accumulation)

    for epoch in range(1, EPOCHS + 1):  # loop over epochs starting at 1
        model.train()  # set model to training mode
        running = 0.0  # running sum of training loss (for display)
        loop = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)  # progress bar over training batches

        optimizer.zero_grad(set_to_none=True)  # clear gradients before starting the epoch

        for step, (input_ids, attn_mask, labels) in enumerate(loop, start=1):  # iterate over each batch
            input_ids = input_ids.to(device, non_blocking=True)  # move batch to device
            attn_mask = attn_mask.to(device, non_blocking=True)  # move batch to device
            labels    = labels.to(device, non_blocking=True)  # move batch to device

            with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):  # enable AMP on GPU
                out = model(input_ids, attention_mask=attn_mask, labels=labels)  # forward pass
                loss = out.loss / GRAD_ACCUM  # divide by accumulation steps so totals line up

            scaler.scale(loss).backward()  # backprop scaled loss (AMP‑safe)

            if step % GRAD_ACCUM == 0:  # every GRAD_ACCUM steps, do an optimizer update
                scaler.unscale_(optimizer)  # unscale gradients for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)  # clip gradients to avoid explosions
                scaler.step(optimizer)  # apply the optimizer step with scaled grads
                scaler.update()  # update the scaler for next iteration
                optimizer.zero_grad(set_to_none=True)  # reset gradients after the step
                scheduler.step()  # advance the learning rate schedule
                global_step += 1  # count this optimizer step

            running += loss.item() * GRAD_ACCUM  # track unscaled loss for logging
            loop.set_postfix(loss=f"{running/step:.4f}")  # update progress bar with avg loss so far

        # ----- end epoch: eval -----
        val_loss, val_ppl = evaluate(model, val_loader)  # run validation to see how we’re doing
        print(f"\nEpoch {epoch} | train_loss={running/len(train_loader):.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")  # log metrics

        # Save best & last
        torch.save(model.state_dict(), LAST_CKPT)  # always save the latest checkpoint
        if val_loss < best_val:  # if this epoch is the best so far
            best_val = val_loss  # update our best score
            torch.save(model.state_dict(), SAVE_BEST_TO)  # save best model weights
            print(f"✅ New best saved to {SAVE_BEST_TO}")  # confirm best save
            with open("best_loss.txt", "w") as f:
                f.write(str(best_val))

    print("Training complete.")  # done with all epochs

    # Optional final test evaluation
    test_path = DATA_DIR / f"{DD_TEST}.txt"  # path to the optional test file
    if test_path.exists():  # only evaluate if test split exists
        test_pairs = load_pairs(DD_TEST, DATA_DIR)  # load test pairs
        test_loader = DataLoader(ChatDataset(test_pairs), batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=True)  # make test loader
        test_loss, test_ppl = evaluate(model, test_loader)  # run test eval
        print(f"Test: loss={test_loss:.4f} | ppl={test_ppl:.2f}")  # print test metrics


# ------------------------ GENERATION ------------------------ #
def _build_condition_tokens(act: Optional[str]=None, emo: Optional[str]=None, topic: Optional[str]=None) -> str:
    toks = []  # collect any condition tokens we want to apply
    if USE_ACT and act:   toks.append(f"<act={act}>")  # add act token if requested
    if USE_EMO and emo:   toks.append(f"<emo={emo}>")  # add emotion token if requested
    if USE_TOPIC and topic is not None: toks.append(f"<topic={topic}>")  # add topic token if requested
    return " ".join(toks)  # concatenate into a single string


def clean_bot_reply(full_text: str, prefix: str) -> str:
    # pull just the Bot: segment (same as you had)
    bot_marker = "Bot:"
    start = full_text.rfind(bot_marker)
    if start == -1:
        out = full_text.replace(prefix, "", 1).strip()
    else:
        out = full_text[start + len(bot_marker):].strip()
    nxt = out.find("User:")
    if nxt != -1:
        out = out[:nxt].strip()

    # small safety: collapse spaces
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def generate_text(prompt: str, max_new_tokens=64,
                  act: Optional[str]=None, emo: Optional[str]=None, topic: Optional[str]=None):
    model.eval()
    cond = _build_condition_tokens(act, emo, topic)
    cond = ((" " + cond) if cond else "")
    prefix = f"User: {prompt} {tokenizer.eos_token} Bot:{cond}"

    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=False).to(device)

    bad_words_ids = tokenizer(["User:"], add_special_tokens=False).input_ids

    with torch.no_grad():
        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,       # allow longer replies
            min_new_tokens=48,        # avoid ultra-short answers
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )

    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    return clean_bot_reply(decoded, prefix).strip()


# ------------------------ MAIN ------------------------ #
if __name__ == "__main__":  # run this block only when executing the file directly
    # If your files are under /mnt/data, set DATA_DIR = Path("/mnt/data") at the top.  # reminder for path changes
    train()  # kick off training
    # Example generations (conditioned)
    print("\nSample (neutral, question, topic 3):",  # show one sample generation with controls
    generate_text("How are you doing today?",  # the user prompt
    act="question", emo="neutral", topic="3"))  # control tokens for the sample
