# assistant/frontier_backend.py
"""Frontier assistant backed by a hosted, OpenAI-compatible API."""
import os
from openai import OpenAI
from .config import PROVIDERS, FRONTIER_PROVIDER, TEMPERATURE, MAX_NEW_TOKENS
import time
from openai import RateLimitError

class FrontierBackend:
    def __init__(self, provider: str = FRONTIER_PROVIDER):
        cfg = PROVIDERS[provider]
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set {cfg['key_env']} in your .env file."
            )
        self.model = cfg["model"]
        print(f"[FrontierBackend] provider base_url={cfg['base_url']} model={self.model}")
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    def chat(self, messages: list[dict], max_retries: int = 6) -> str:
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                if "PerDay" in str(e) or attempt == max_retries - 1:
                    raise            # daily cap won't reset by waiting
                wait = 35
                print(f"  (rate limited; waiting {wait}s, attempt {attempt+1})")
                time.sleep(wait)