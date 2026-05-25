# assistant/oss_backend.py
"""Open-source assistant: Qwen2.5 run locally via Hugging Face transformers."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import OSS_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class OSSBackend:
    def __init__(self, model_name: str = OSS_MODEL):
        self.model_name = model_name
        print(f"Loading OSS model '{model_name}' (first run downloads weights)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",          # change to "auto" if you have a GPU
        )
        print("OSS model loaded.")

    def chat(self, messages: list[dict]) -> str:
        # Qwen expects a chat template; transformers builds the prompt for us.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip the prompt tokens; keep only the newly generated answer.
        new_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()