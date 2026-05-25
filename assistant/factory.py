# assistant/factory.py
"""Return a backend by name. Both backends expose .chat(messages) -> str."""
from .config import DEFAULT_BACKEND


def get_backend(name: str = DEFAULT_BACKEND):
    if name == "oss":
        from .oss_backend import OSSBackend
        return OSSBackend()
    elif name == "frontier":
        from .frontier_backend import FrontierBackend
        return FrontierBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'oss' or 'frontier'.")