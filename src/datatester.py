# === Testing + Web UI in one file ============================================
# Your original testing code, plus a FastAPI website that calls generate_text().
# Run EITHER the CLI chat (python model_runtime.py) OR the web server:
#   uvicorn model_runtime:app --reload --port 8000

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import Optional
import re

# --- NEW: Web server bits ---
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Tokenizer ----------------
tokenizer = GPT2LMHeadModel  # placeholder to satisfy type hints in some editors
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ==== REBUILD TRAIN-TIME SPECIAL TOKENS EXACTLY ====
USE_ACT   = True
USE_EMO   = True
USE_TOPIC = True

ACT_MAP = {1: "inform", 2: "question", 3: "directive", 4: "commissive"}
EMO_MAP = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}

specials = []
if USE_ACT:
    specials += [f"<act={name}>" for name in sorted(set(ACT_MAP.values()))]
if USE_EMO:
    specials += [f"<emo={name}>" for name in sorted(set(EMO_MAP.values()))]
if USE_TOPIC:
    specials += [f"<topic={i}>" for i in range(0, 20)]
if specials:
    tokenizer.add_special_tokens({"additional_special_tokens": specials})

# ---------------- Model ----------------
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Load your finetuned weights
ckpt_path = "chatbot_gpt2_best.pth"
try:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
except TypeError:
    # Older PyTorch doesn't support weights_only
    state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# ---------------- Generation utils ----------------
def _build_condition_tokens(act: Optional[str]=None, emo: Optional[str]=None, topic: Optional[str]=None) -> str:
    toks = []
    if USE_ACT and act:   toks.append(f"<act={act}>")
    if USE_EMO and emo:   toks.append(f"<emo={emo}>")
    if USE_TOPIC and topic is not None: toks.append(f"<topic={topic}>")
    return " ".join(toks)

def clean_bot_reply(full_text: str, prefix: str) -> str:
    bot_marker = "Bot:"
    start = full_text.rfind(bot_marker)
    if start == -1:
        out = full_text.replace(prefix, "", 1).strip()
    else:
        out = full_text[start + len(bot_marker):].strip()
    nxt = out.find("User:")
    if nxt != -1:
        out = out[:nxt].strip()
    m = re.search(r'([.!?])(\s|$)', out)
    if m:
        out = out[:m.end()].strip()
    out = re.sub(r'\s+', ' ', out).strip()
    return out

def generate_text(prompt: str, max_new_tokens=64,
                  act: Optional[str]=None, emo: Optional[str]=None, topic: Optional[str]=None):
    model.eval()
    cond = _build_condition_tokens(act, emo, topic)
    cond = ((" " + cond) if cond else "")
    prefix = f"User: {prompt} {tokenizer.eos_token} Bot:{cond}"

    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=128, padding=False).to(device)
    bad_words_ids = tokenizer(["User:"], add_special_tokens=False).input_ids

    with torch.no_grad():
        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )

    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    return clean_bot_reply(decoded, prefix).strip()

# ================== WEB APP ==================
app = FastAPI()

_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Transformer Chat</title>
<style>
  :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  body { margin: 0; background: #f6f7f9; }
  .wrap { max-width: 820px; margin: 40px auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); }
  header { padding: 16px 20px; border-bottom: 1px solid #eee; display:flex; justify-content:space-between; align-items:center; }
  h1 { font-size: 18px; margin: 0; }
  .chat { height: 60vh; overflow-y: auto; padding: 16px 20px; }
  .bubble { max-width: 80%; padding: 10px 14px; border-radius: 14px; margin-bottom: 10px; white-space: pre-wrap; word-break: break-word; }
  .user { margin-left: auto; background: #111827; color: #fff; }
  .bot { margin-right: auto; background: #f3f4f6; color: #111; }
  footer { padding: 12px 20px; border-top: 1px solid #eee; }
  textarea { width: 100%; resize: vertical; min-height: 56px; max-height: 200px; padding: 12px; border: 1px solid #d1d5db; border-radius: 12px; box-sizing: border-box; }
  .row { display:flex; gap: 10px; align-items: stretch; margin-top: 8px; }
  button { padding: 12px 16px; border: none; border-radius: 12px; background: #111827; color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { background: #e5e7eb; color: #9ca3af; cursor: not-allowed; }
  .small { font-size: 12px; color: #6b7280 }
  .controls { display:flex; gap:8px; flex-wrap:wrap; margin: 8px 0 0; }
  .controls input, .controls select { padding: 8px; border:1px solid #d1d5db; border-radius:10px; }
</style>
</head><body>
  <div class="wrap">
    <header>
      <h1>Transformer Chat</h1>
      <div class="small">Served by FastAPI (no Node.js needed)</div>
    </header>

    <div id="chat" class="chat"></div>

    <footer>
      <label class="small">System prompt</label>
      <textarea id="sys" placeholder="You are a helpful assistant."></textarea>
      <div class="controls">
        <label>Act:
          <select id="act">
            <option value="">(none)</option>
            <option>inform</option>
            <option>question</option>
            <option>directive</option>
            <option>commissive</option>
          </select>
        </label>
        <label>Emotion:
          <select id="emo">
            <option value="">(none)</option>
            <option>neutral</option>
            <option>anger</option>
            <option>disgust</option>
            <option>fear</option>
            <option>happiness</option>
            <option>sadness</option>
            <option>surprise</option>
          </select>
        </label>
        <label>Topic:
          <input id="topic" type="text" placeholder="e.g., 3" style="width:80px"/>
        </label>
      </div>
      <div class="row">
        <textarea id="input" placeholder="Type a message..."></textarea>
        <button id="send" disabled>Send</button>
      </div>
    </footer>
  </div>

<script>
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');
  const sys = document.getElementById('sys');
  const act = document.getElementById('act');
  const emo = document.getElementById('emo');
  const topic = document.getElementById('topic');
  const sendBtn = document.getElementById('send');

  function addBubble(text, who) {
    const div = document.createElement('div');
    div.className = 'bubble ' + (who === 'user' ? 'user' : 'bot');
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function setSending(on) {
    sendBtn.disabled = on || !input.value.trim();
    input.disabled = on; sys.disabled = on; act.disabled = on; emo.disabled = on; topic.disabled = on;
  }

  input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !sendBtn.disabled) {
      e.preventDefault(); sendBtn.click();
    }
  });

  sendBtn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return;
    addBubble(text, 'user');
    input.value = '';
    setSending(true);
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          system: sys.value || null,
          act: act.value || null,
          emo: emo.value || null,
          topic: topic.value || null
        })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      addBubble(data.reply ?? String(data), 'bot');
    } catch (e) {
      addBubble('Error: ' + e.message, 'bot');
    } finally {
      setSending(false); input.focus();
    }
  });

  sendBtn.disabled = !input.value.trim();
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML

class ChatIn(BaseModel):
    text: str
    system: str | None = None
    act: str | None = None
    emo: str | None = None
    topic: str | None = None

@app.post("/chat")
def chat(body: ChatIn):
    # (We ignore 'system' here because your generate_text formats only User/Bot;
    # if you want system conditioning, prepend to prompt inside generate_text.)
    reply = generate_text(
        body.text,
        act=body.act or None,
        emo=body.emo or None,
        topic=body.topic or None,
    )
    return JSONResponse({"reply": reply})

# ================== OPTIONAL: CLI mode ==================
if __name__ == "__main__":
    # Simple terminal chat loop (unchanged behavior)
    try:
        while True:
            prompt = input("You: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            response = generate_text(prompt, act="question", emo="neutral", topic="3")
            print("Bot:", response)
    except KeyboardInterrupt:
        pass
