# app.py
"""Gradio web UI. Pipeline: guardrail -> tool -> model -> guardrail."""
import gradio as gr
from dotenv import load_dotenv

from assistant.memory import ConversationMemory
from assistant.factory import get_backend
from assistant.guardrails import check_input, check_output
from assistant.tools import maybe_use_tool
from assistant.config import ENABLE_GUARDRAILS

load_dotenv()

# Cache backends so we don't reload Qwen on every message.
_BACKENDS = {}


def get_cached_backend(name: str):
    if name not in _BACKENDS:
        _BACKENDS[name] = get_backend(name)
    return _BACKENDS[name]


def respond(message, history, backend_name, memory):
    if memory is None:
        memory = ConversationMemory()

    # 1) Input guardrail
    if ENABLE_GUARDRAILS:
        allowed, reason = check_input(message)
        if not allowed:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reason},
            ]
            return history, memory, ""

    # 2) Tool use (skips the model when a tool clearly applies)
    tool_result = maybe_use_tool(message)
    if tool_result is not None:
        reply = tool_result
    else:
        # 3) Model call with memory
        memory.add_user(message)
        backend = get_cached_backend(backend_name)
        reply = backend.chat(memory.build_context())
        memory.add_assistant(reply)

    # 4) Output guardrail
    if ENABLE_GUARDRAILS:
        reply = check_output(reply)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return history, memory, ""


def reset_all():
    return [], ConversationMemory(), ""


with gr.Blocks(title="Dual Personal Assistant") as demo:
    gr.Markdown("# Personal Assistant — OSS vs Frontier\nSwitch backends and chat. Includes guardrails + tools.")

    backend_dropdown = gr.Dropdown(
        choices=["frontier", "oss"], value="frontier", label="Backend"
    )
    chatbot = gr.Chatbot(height=420)
    memory_state = gr.State(ConversationMemory())

    msg = gr.Textbox(placeholder="Ask me anything...", label="Message")
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")

    send.click(respond, [msg, chatbot, backend_dropdown, memory_state],
               [chatbot, memory_state, msg])
    msg.submit(respond, [msg, chatbot, backend_dropdown, memory_state],
               [chatbot, memory_state, msg])
    clear.click(reset_all, None, [chatbot, memory_state, msg])

if __name__ == "__main__":
    demo.launch()