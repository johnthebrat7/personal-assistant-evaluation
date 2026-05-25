# space/app.py
"""Public OSS personal assistant (Qwen2.5-0.5B) with guardrails, tools,
short-term memory, and lightweight observability logging."""
import ast
import json
import operator
import re
import time
from datetime import datetime, timezone

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_TURNS = 8
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
LOG_PATH = "logs.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise personal assistant. "
    "Answer clearly and admit when you do not know something rather than guessing. "
    "Refuse requests that are illegal, harmful, or unsafe, and explain briefly why."
)

# ---------- Load model once at startup ----------
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="cpu"
)
print("Model loaded.")

# ---------- Guardrails ----------
BLOCKED_PATTERNS = [
    r"\bhow to (make|build|create).{0,20}(bomb|explosive|weapon)\b",
    r"\b(synthesize|make).{0,15}(meth|methamphetamine|fentanyl)\b",
    r"\bhow to (kill|murder|poison)\b",
    r"\b(child|minor).{0,15}(sexual|porn)\b",
    r"\bpick a lock.{0,20}(break|burglar)\b",
]
REFUSAL = ("I can't help with that — it appears to involve harmful or unsafe "
           "activity. If you have a safe, legitimate version, please rephrase.")


def check_input(text: str):
    low = text.lower()
    for p in BLOCKED_PATTERNS:
        if re.search(p, low):
            return False, REFUSAL
    return True, ""


def check_output(text: str):
    low = text.lower()
    for p in BLOCKED_PATTERNS:
        if re.search(p, low):
            return REFUSAL
    return text

# ---------- Tools ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported")


def maybe_use_tool(text: str):
    low = text.lower()
    if any(w in low for w in ["what time", "what's the date", "today's date",
                              "current date", "what day"]):
        now = datetime.now()
        return f"The current date and time is {now.strftime('%A, %d %B %Y, %H:%M')}."
    m = re.search(r"[-+]?\d[\d\s+\-*/().%^]*\d", text)
    if m and any(op in text for op in "+-*/^%"):
        try:
            return f"The answer is {_safe_eval(ast.parse(m.group().replace('^','**'), mode='eval').body)}."
        except Exception:
            return "I couldn't evaluate that expression."
    return None

# ---------- Model generation ----------
def generate(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0, pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip()

# ---------- Observability ----------
def log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass  # never let logging crash the app


def fresh_stats():
    return {"turns": 0, "blocked": 0, "tool": 0, "total_latency": 0.0}


def stats_md(s):
    avg = (s["total_latency"] / s["turns"]) if s["turns"] else 0
    return (f"**Turns:** {s['turns']}  |  **Guardrail blocks:** {s['blocked']}  "
            f"|  **Tool uses:** {s['tool']}  |  **Avg latency:** {avg:.1f}s")

# ---------- Chat handler ----------
def respond(message, history, mem, stats):
    history = history or []
    mem = mem or []
    stats = stats or fresh_stats()
    if not message.strip():
        return history, mem, stats, stats_md(stats), ""

    route = "model"
    start = time.time()

    allowed, reason = check_input(message)
    if not allowed:
        reply, route = reason, "blocked"
        stats["blocked"] += 1
    else:
        tool = maybe_use_tool(message)
        if tool is not None:
            reply, route = tool, "tool"
            stats["tool"] += 1
        else:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            msgs += mem[-2 * MEMORY_TURNS:]
            msgs += [{"role": "user", "content": message}]
            reply = check_output(generate(msgs))
            mem = mem + [{"role": "user", "content": message},
                         {"role": "assistant", "content": reply}]

    latency = round(time.time() - start, 2)
    stats["turns"] += 1
    stats["total_latency"] += latency
    log_event({"route": route, "input": message[:300],
               "output": reply[:300], "latency_s": latency})

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": reply}]
    return history, mem, stats, stats_md(stats), ""


def reset_all():
    s = fresh_stats()
    return [], [], s, stats_md(s), ""

# ---------- UI ----------
with gr.Blocks(title="Qwen Personal Assistant") as demo:
    gr.Markdown("# 🤖 Qwen2.5 Personal Assistant\n"
                "Open-source assistant with guardrails, tools, memory & observability.")
    chatbot = gr.Chatbot(height=420)
    stats_box = gr.Markdown(stats_md(fresh_stats()))
    mem_state = gr.State([])
    stats_state = gr.State(fresh_stats())
    msg = gr.Textbox(placeholder="Ask me anything... (try: what is 23*47)", label="Message")
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    inputs = [msg, chatbot, mem_state, stats_state]
    outputs = [chatbot, mem_state, stats_state, stats_box, msg]
    send.click(respond, inputs, outputs)
    msg.submit(respond, inputs, outputs)
    clear.click(reset_all, None, outputs)

if __name__ == "__main__":
    demo.launch()