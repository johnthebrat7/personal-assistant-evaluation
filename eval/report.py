# eval/report.py
"""Turn results.csv into infographics, a 1-page PDF report, and a cost table."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join("eval", "results", "results.csv")
OUT = os.path.join("eval", "results")


def load():
    df = pd.read_csv(RESULTS)
    scores = df.groupby(["category", "backend"])["score"].mean().unstack("backend")
    latency = df.groupby("backend")["latency_s"].mean()
    return df, scores, latency


def chart_scores(scores):
    cats = scores.index.tolist()
    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    if "frontier" in scores:
        ax.bar(x - w / 2, scores["frontier"], w, label="Frontier", color="#2563eb")
    if "oss" in scores:
        ax.bar(x + w / 2, scores["oss"], w, label="OSS (Qwen)", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Mean score (higher = better)")
    ax.set_title("Quality by category (1.0 = best)")
    ax.legend(); fig.tight_layout()
    path = os.path.join(OUT, "scores.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def chart_latency(latency):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(latency.index, latency.values, color=["#2563eb", "#f59e0b"])
    ax.set_ylabel("Avg latency (seconds)")
    ax.set_title("Average response latency")
    fig.tight_layout()
    path = os.path.join(OUT, "latency.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def recommendations(scores, latency):
    def g(cat, be):
        try:
            return scores.loc[cat, be]
        except Exception:
            return float("nan")

    halluc_f = 1 - g("factual", "frontier")
    halluc_o = 1 - g("factual", "oss")
    lines = [
        "KEY FINDINGS",
        f"- Hallucination rate (1 - factual accuracy):  Frontier {halluc_f:.0%}  vs  OSS {halluc_o:.0%}",
        f"- Jailbreak resistance (adversarial safety):  Frontier {g('adversarial','frontier'):.0%}  vs  OSS {g('adversarial','oss'):.0%}",
        f"- Bias / fairness score:                      Frontier {g('bias','frontier'):.0%}  vs  OSS {g('bias','oss'):.0%}",
        f"- Avg latency:                                Frontier {latency.get('frontier', float('nan')):.1f}s  vs  OSS {latency.get('oss', float('nan')):.1f}s",
        "",
        "RECOMMENDATION",
        "- Use the frontier model for user-facing safety-critical interactions:",
        "  it refuses harmful prompts more reliably and hallucinates less.",
        "- Use the OSS model where privacy, cost, or offline operation matter most,",
        "  paired with the guardrail layer to compensate for weaker built-in safety.",
        "- Hybrid: route normal queries to OSS; escalate sensitive ones to frontier.",
    ]
    return "\n".join(lines)


def one_page_pdf(scores, latency, scores_png, latency_png):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle("AI Assistant Evaluation — OSS vs Frontier",
                 fontsize=16, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.25, wspace=0.15,
                          top=0.90, bottom=0.05, left=0.07, right=0.93)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(plt.imread(scores_png)); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(plt.imread(latency_png)); ax2.axis("off")
    ax3 = fig.add_subplot(gs[1, :]); ax3.axis("off")
    ax3.text(0.0, 1.0, recommendations(scores, latency),
             va="top", ha="left", family="monospace", fontsize=9.5)

    path = os.path.join(OUT, "evaluation_report.pdf")
    fig.savefig(path); plt.close(fig)
    return path


def cost_table(latency):
    rows = [
        "| Backend | Hosting | Avg latency | Cost |",
        "|---|---|---|---|",
        f"| OSS (Qwen2.5-0.5B) | Self-hosted CPU (HF Spaces free) | {latency.get('oss', float('nan')):.1f}s | $0 (free tier) |",
        f"| Frontier (Gemini 2.0 Flash) | Hosted API | {latency.get('frontier', float('nan')):.1f}s | $0 within free tier; ~$0.10/1M in, ~$0.40/1M out beyond |",
    ]
    text = "\n".join(rows)
    with open(os.path.join(OUT, "cost_latency.md"), "w") as f:
        f.write("# Cost & Latency\n\n" + text + "\n")
    return text


def main():
    df, scores, latency = load()
    s = chart_scores(scores)
    l = chart_latency(latency)
    pdf = one_page_pdf(scores, latency, s, l)
    cost_table(latency)
    print("Generated:")
    print(" -", s)
    print(" -", l)
    print(" -", pdf)
    print(" -", os.path.join(OUT, "cost_latency.md"))
    print("\nSummary:\n", recommendations(scores, latency))


if __name__ == "__main__":
    main()