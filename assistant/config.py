# assistant/config.py
"""Central configuration. Everything tunable lives here."""
import os

# Which frontier provider to use. All expose an OpenAI-compatible endpoint.
FRONTIER_PROVIDER = os.getenv("FRONTIER_PROVIDER", "gemini")

# ----- Open-source model (Assignment requirement 1) -----
OSS_MODEL = os.getenv("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Which backend the app uses by default: "oss" or "frontier"
DEFAULT_BACKEND = os.getenv("ASSISTANT_BACKEND", "frontier")

ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": os.getenv("FRONTIER_MODEL", "anthropic/claude-3.5-sonnet"),
        "key_env": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": os.getenv("FRONTIER_MODEL", "gemini-2.5-flash"),
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": os.getenv("FRONTIER_MODEL", "llama-3.3-70b-versatile"),
        "key_env": "GROQ_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": os.getenv("FRONTIER_MODEL", "deepseek-chat"),
        "key_env": "DEEPSEEK_API_KEY",
    },
}

# Short-term memory: number of recent turns (1 turn = user + assistant) to keep.
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))

# Generation settings
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

SYSTEM_PROMPT = (
    "You are a helpful, honest, and concise personal assistant. "
    "Answer clearly and admit when you do not know something rather than guessing. "
    "Refuse requests that are illegal, harmful, or unsafe, and explain briefly why."
)