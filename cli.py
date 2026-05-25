# cli.py
"""Terminal chat loop with selectable backend (oss | frontier)."""
import sys
from dotenv import load_dotenv
from assistant.memory import ConversationMemory
from assistant.factory import get_backend

load_dotenv()


def main():
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "frontier"
    print(f"Backend: {backend_name}")
    backend = get_backend(backend_name)
    memory = ConversationMemory()
    print("Assistant ready. Type 'quit' to exit, 'reset' to clear memory.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "reset":
            memory.reset()
            print("(memory cleared)\n")
            continue
        if not user:
            continue

        memory.add_user(user)
        reply = backend.chat(memory.build_context())
        memory.add_assistant(reply)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()