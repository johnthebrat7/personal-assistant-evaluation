# eval/run_eval.py
"""Run both backends over all prompt sets, score with the judge, save CSV."""
import csv
import json
import os
import time
from dotenv import load_dotenv

from assistant.factory import get_backend
from assistant.memory import ConversationMemory
from eval.judge import judge

load_dotenv()

CATEGORIES = ["factual", "adversarial", "bias"]
BACKENDS = ["oss","frontier"]
PROMPT_DIR = os.path.join("eval", "prompts")
RESULTS_PATH = os.path.join("eval", "results", "results.csv")


def load_prompts(category):
    with open(os.path.join(PROMPT_DIR, f"{category}.json")) as f:
        return json.load(f)


def _done_keys():
    """Keys already in results.csv so we can resume without redoing them."""
    done = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["backend"], row["category"], row["id"]))
    return done


def run():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    done = _done_keys()
    fields = ["backend", "category", "id", "prompt", "answer",
              "score", "reason", "latency_s"]
    new_file = not os.path.exists(RESULTS_PATH)

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()

        for backend_name in BACKENDS:
            print(f"\n=== Backend: {backend_name} ===")
            backend = get_backend(backend_name)

            for category in CATEGORIES:
                for item in load_prompts(category):
                    key = (backend_name, category, item["id"])
                    if key in done:
                        print(f"  skip {key} (already done)")
                        continue

                    memory = ConversationMemory()
                    memory.add_user(item["prompt"])
                    start = time.time()
                    try:
                        answer = backend.chat(memory.build_context())
                    except Exception as e:
                        print(f"  STOPPING: {backend_name} hit an error "
                              f"(likely daily quota). Resume later. [{e}]")
                        return
                    latency = round(time.time() - start, 2)

                    verdict = judge(category, item["prompt"], answer,
                                    item.get("expected"))
                    print(f"[{backend_name}/{category}/{item['id']}] "
                          f"score={verdict['score']} ({latency}s)")

                    writer.writerow({
                        "backend": backend_name, "category": category,
                        "id": item["id"], "prompt": item["prompt"],
                        "answer": answer.replace("\n", " ")[:300],
                        "score": verdict["score"],
                        "reason": verdict["reason"][:200],
                        "latency_s": latency,
                    })
                    f.flush()           # persist immediately
                    time.sleep(2)       # Groq is fast; small spacing is enough
    print(f"\nDone. Results in {RESULTS_PATH}")

if __name__ == "__main__":
    run()