"""
Evaluate LLM-generated text highlighting against manual highlight annotations.

This script evaluates whether the LLM highlights IMPORTANT transcript parts.
It compares predicted highlight quotes against manually annotated gold phrases.

Core definitions per result:
  - gold positive      := at least one gold highlight phrase exists
  - predicted positive := the LLM returned at least one quote
  - matched            := at least one predicted quote matches a gold phrase

Metrics mirror the retrieval evaluator shape:
  - Highlight Precision@K / Recall@K / F1@K for K in {10, 20, 30, 40, 50}
  - Per-query precision-recall curve at rank 1..N
  - Macro-averaged curves and per-query plots

Required inputs:
  - query_results.txt            (output from getRankings.py)
  - annotated_highlights.txt     (manual highlight annotations)

LLM predictions are cached in results/highlight_predictions.json to avoid
re-calling the model on every run.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from google import genai

from evaluate_metrics import K_VALUES


QUERY_RESULTS_FILE = "query_results.txt"
HIGHLIGHT_ANNOTATED_FILE = "annotated_highlights.txt"
RESULTS_DIR = "results"

PREDICTIONS_FILE = f"{RESULTS_DIR}/highlight_predictions.json"
PER_QUERY_PLOT_DIR = f"{RESULTS_DIR}/highlight_pr_curves_per_query"
AVG_PLOT_PATH = f"{RESULTS_DIR}/highlight_pr_curve.png"
PER_QUERY_METRICS_PATH = f"{RESULTS_DIR}/highlight_metrics_per_query.csv"
AVERAGED_METRICS_PATH = f"{RESULTS_DIR}/highlight_metrics_averaged.csv"
PR_CURVE_PATH = f"{RESULTS_DIR}/highlight_pr_curve_per_query.csv"

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_SIZE = 5
MATCH_THRESHOLD = 0.82
NO_HIGHLIGHTS_TOKEN = "NONE"


def parse_query_results(filepath: str) -> dict[str, list[dict[str, Any]]]:
    """
    Parse query_results.txt from getRankings.py.

    Expected pattern per query:
      <query line>
      <num_found>
      [1]
      Podcast : ...
      Episode : ...
      Content : ...
      Time    : ...
      ...

    Returns:
      {
        query: [
          {"rank": int, "show": str, "episode": str, "text": str, "time": str},
          ...
        ],
        ...
      }
    """
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    data: dict[str, list[dict[str, Any]]] = {}

    i = 0
    while i < len(lines):
        query = lines[i].strip()
        if not query:
            i += 1
            continue

        if i + 1 >= len(lines):
            break
        if not re.fullmatch(r"\d+", lines[i + 1].strip()):
            i += 1
            continue

        i += 2
        results: list[dict[str, Any]] = []

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if not re.fullmatch(r"\[\d+\]", line):
                if i + 1 < len(lines) and re.fullmatch(r"\d+", lines[i + 1].strip()):
                    break
                i += 1
                continue

            rank = int(line.strip("[]"))
            podcast_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            episode_line = lines[i + 2].strip() if i + 2 < len(lines) else ""
            content_line = lines[i + 3].strip() if i + 3 < len(lines) else ""
            time_line = lines[i + 4].strip() if i + 4 < len(lines) else ""

            show = (
                podcast_line.split("Podcast :", 1)[-1].strip()
                if "Podcast :" in podcast_line
                else podcast_line
            )
            episode = (
                episode_line.split("Episode :", 1)[-1].strip()
                if "Episode :" in episode_line
                else episode_line
            )
            text = (
                content_line.split("Content :", 1)[-1].strip()
                if "Content :" in content_line
                else content_line
            )
            time = (
                time_line.split("Time    :", 1)[-1].strip()
                if "Time    :" in time_line
                else time_line
            )

            results.append(
                {
                    "rank": rank,
                    "show": show,
                    "episode": episode,
                    "text": text,
                    "time": time,
                }
            )
            i += 5

        data[query] = results

    return data


def parse_annotated_highlights(
    filepath: str, known_queries: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """
    Parse annotated_highlights.txt.

    Format (mirrors annotated_results.txt style):
      <query>
      <show>
      <episode>
      <highlight_1 || highlight_2 || ... || highlight_n>
      <blank line>

    Use "NONE" when no phrase in the chunk should be highlighted.

    Returns:
      {
        query: [
          {"show": str, "episode": str, "gold_highlights": list[str]},
          ...
        ]
      }
    """
    data: dict[str, list[dict[str, Any]]] = {}
    current_query: str | None = None
    current_results: list[dict[str, Any]] = []

    lines = Path(filepath).read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Query boundaries are detected from known query strings.
        if line in known_queries and current_query != line:
            if current_query is not None and current_results:
                data[current_query] = current_results
            current_query = line
            current_results = []
            i += 1
            continue

        if current_query is not None:
            show = line

            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            episode = lines[i].strip() if i < len(lines) else ""

            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            highlights_line = lines[i].strip() if i < len(lines) else ""

            if show and episode and highlights_line:
                if highlights_line.upper() == NO_HIGHLIGHTS_TOKEN:
                    gold_highlights: list[str] = []
                else:
                    gold_highlights = [
                        p.strip() for p in highlights_line.split("||") if p.strip()
                    ]

                current_results.append(
                    {
                        "show": show,
                        "episode": episode,
                        "gold_highlights": gold_highlights,
                    }
                )

                i += 1
                continue

        i += 1

    if current_query is not None and current_results:
        data[current_query] = current_results

    return data


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return key


def _build_prompt(query: str, batch: list[dict[str, Any]]) -> str:
    lines = [
        "You are given a query and transcript chunks.",
        "Extract the most relevant SHORT phrases from each chunk.",
        "",
        "Rules:",
        "1. Copy phrases EXACTLY from the text.",
        "2. Do NOT paraphrase.",
        "3. Return 1-3 phrases per result.",
        "4. Each phrase should be 5-15 words.",
        "5. If not relevant, return empty list.",
        "",
        f"Query: {query}",
        "",
        "Results:",
    ]

    for idx, item in enumerate(batch):
        lines.extend([f"Result {idx}:", "Text:", item["text"], ""])

    return "\n".join(lines)


def _extract_batch_quotes(
    client: genai.Client,
    model_name: str,
    query: str,
    batch: list[dict[str, Any]],
) -> dict[int, list[str]]:
    schema = {
        "type": "object",
        "properties": {
            "highlights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "result_index": {"type": "integer"},
                        "quotes": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["result_index", "quotes"],
                },
            }
        },
        "required": ["highlights"],
    }

    response = client.models.generate_content(
        model=model_name,
        contents=_build_prompt(query, batch),
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        },
    )

    payload = (response.text or "").strip()
    if not payload:
        return {}

    parsed = json.loads(payload)
    quote_map: dict[int, list[str]] = {}

    for item in parsed.get("highlights", []):
        idx = item.get("result_index")
        quotes = item.get("quotes", [])
        if not isinstance(idx, int) or not isinstance(quotes, list):
            continue
        quote_map[idx] = [q.strip() for q in quotes if isinstance(q, str) and q.strip()]

    return quote_map


def generate_predictions(
    query_results: dict[str, list[dict[str, Any]]],
    model_name: str,
    batch_size: int,
    max_queries: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {query: [{rank, show, episode, quotes, predicted_positive}, ...]}"""
    client = genai.Client(api_key=_api_key())
    predictions: dict[str, list[dict[str, Any]]] = {}

    query_items = list(query_results.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    for q_idx, (query, results) in enumerate(query_items, start=1):
        print(f"[{q_idx}/{len(query_items)}] Generating highlights for query: {query}")
        rows: list[dict[str, Any]] = []

        for start in range(0, len(results), batch_size):
            batch = results[start : start + batch_size]
            quote_map = _extract_batch_quotes(client, model_name, query, batch)

            for local_idx, item in enumerate(batch):
                quotes = quote_map.get(local_idx, [])
                rows.append(
                    {
                        "rank": item["rank"],
                        "show": item["show"],
                        "episode": item["episode"],
                        "quotes": quotes,
                        "predicted_positive": bool(quotes),
                    }
                )

        predictions[query] = rows

    return predictions


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _phrase_match(a: str, b: str) -> bool:
    a_n = _norm(a)
    b_n = _norm(b)
    if not a_n or not b_n:
        return False

    # Strong exact/substring signals first.
    if a_n == b_n:
        return True
    if a_n in b_n or b_n in a_n:
        return True

    return difflib.SequenceMatcher(None, a_n, b_n).ratio() >= MATCH_THRESHOLD


def _has_match(pred_quotes: list[str], gold_quotes: list[str]) -> bool:
    if not pred_quotes or not gold_quotes:
        return False

    for p in pred_quotes:
        for g in gold_quotes:
            if _phrase_match(p, g):
                return True
    return False


def precision_at_k_binary(pred_hit: list[bool], pred_pos: list[bool], k: int) -> float:
    k = min(k, len(pred_hit), len(pred_pos))
    if k <= 0:
        return 0.0

    tp = sum(1 for h in pred_hit[:k] if h)
    pp = sum(1 for p in pred_pos[:k] if p)
    if pp == 0:
        return 0.0
    return tp / pp


def recall_at_k_binary(pred_hit: list[bool], gold_pos: list[bool], k: int) -> float:
    k = min(k, len(pred_hit), len(gold_pos))
    if k <= 0:
        return 0.0

    tp = sum(1 for h in pred_hit[:k] if h)
    total_pos = sum(1 for g in gold_pos if g)
    if total_pos == 0:
        return 0.0
    return tp / total_pos


def f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_highlight_metrics(
    annotations: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_query: dict[str, Any] = {}
    avg_p = {k: [] for k in K_VALUES}
    avg_r = {k: [] for k in K_VALUES}
    avg_f1 = {k: [] for k in K_VALUES}

    p_curves: list[list[float]] = []
    r_curves: list[list[float]] = []

    common_queries = [q for q in annotations if q in predictions]
    if not common_queries:
        raise RuntimeError(
            "No overlapping queries between annotations and predictions."
        )

    for query in common_queries:
        ann = annotations[query]
        pred_rows = predictions[query]

        n = min(len(ann), len(pred_rows))
        if n == 0:
            continue

        gold_pos: list[bool] = []
        pred_pos: list[bool] = []
        pred_hit: list[bool] = []

        for i in range(n):
            gold_quotes = ann[i].get("gold_highlights", [])
            pred_quotes = pred_rows[i].get("quotes", [])

            gold_pos.append(bool(gold_quotes))
            pred_pos.append(bool(pred_quotes))
            pred_hit.append(_has_match(pred_quotes, gold_quotes))

        p_at_k = {k: precision_at_k_binary(pred_hit, pred_pos, k) for k in K_VALUES}
        r_at_k = {k: recall_at_k_binary(pred_hit, gold_pos, k) for k in K_VALUES}
        f1_at_k = {k: f1_score(p_at_k[k], r_at_k[k]) for k in K_VALUES}

        per_query[query] = {
            "precision": p_at_k,
            "recall": r_at_k,
            "f1": f1_at_k,
            "n_evaluated": n,
        }

        for k in K_VALUES:
            avg_p[k].append(p_at_k[k])
            avg_r[k].append(r_at_k[k])
            avg_f1[k].append(f1_at_k[k])

        p_curve: list[float] = []
        r_curve: list[float] = []
        for k in range(1, n + 1):
            p_curve.append(precision_at_k_binary(pred_hit, pred_pos, k))
            r_curve.append(recall_at_k_binary(pred_hit, gold_pos, k))
        p_curves.append(p_curve)
        r_curves.append(r_curve)

    averaged = {
        k: {
            "precision": float(np.mean(avg_p[k])) if avg_p[k] else 0.0,
            "recall": float(np.mean(avg_r[k])) if avg_r[k] else 0.0,
            "f1": float(np.mean(avg_f1[k])) if avg_f1[k] else 0.0,
        }
        for k in K_VALUES
    }

    return {
        "queries": common_queries,
        "per_query": per_query,
        "averaged": averaged,
        "p_curves": p_curves,
        "r_curves": r_curves,
    }


def _interp_pr(
    p_curve: list[float], r_curve: list[float], recall_grid: np.ndarray
) -> np.ndarray:
    if not r_curve or r_curve[-1] == 0:
        return np.zeros(len(recall_grid))
    paired = sorted(zip(r_curve, p_curve))
    r_sorted = [x[0] for x in paired]
    p_sorted = [x[1] for x in paired]
    return np.interp(recall_grid, r_sorted, p_sorted)


def save_metrics_per_query(
    results: dict[str, Any], output_path: str = PER_QUERY_METRICS_PATH
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "query",
                "K",
                "highlight_precision",
                "highlight_recall",
                "highlight_f1",
                "n_evaluated",
            ]
        )
        for query in results["queries"]:
            n_eval = results["per_query"][query]["n_evaluated"]
            for k in K_VALUES:
                writer.writerow(
                    [
                        query,
                        k,
                        round(results["per_query"][query]["precision"][k], 6),
                        round(results["per_query"][query]["recall"][k], 6),
                        round(results["per_query"][query]["f1"][k], 6),
                        n_eval,
                    ]
                )


def save_metrics_averaged(
    results: dict[str, Any], output_path: str = AVERAGED_METRICS_PATH
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["K", "avg_highlight_precision", "avg_highlight_recall", "avg_highlight_f1"]
        )
        for k in K_VALUES:
            writer.writerow(
                [
                    k,
                    round(results["averaged"][k]["precision"], 6),
                    round(results["averaged"][k]["recall"], 6),
                    round(results["averaged"][k]["f1"], 6),
                ]
            )


def save_pr_curve_per_query(
    results: dict[str, Any], output_path: str = PR_CURVE_PATH
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query", "rank", "precision", "recall"])
        for query, p_curve, r_curve in zip(
            results["queries"], results["p_curves"], results["r_curves"]
        ):
            for rank, (p, r) in enumerate(zip(p_curve, r_curve), start=1):
                writer.writerow([query, rank, round(p, 6), round(r, 6)])


def plot_avg_precision_recall(
    results: dict[str, Any], output_path: str = AVG_PLOT_PATH
) -> None:
    recall_grid = np.linspace(0, 1, 101)
    interp_precisions = [
        _interp_pr(p, r, recall_grid)
        for p, r in zip(results["p_curves"], results["r_curves"])
    ]
    mean_precision = (
        np.mean(interp_precisions, axis=0)
        if interp_precisions
        else np.zeros_like(recall_grid)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        recall_grid,
        mean_precision,
        color="#16A34A",
        linewidth=2.5,
        label="Mean highlight P-R curve",
    )
    ax.fill_between(recall_grid, mean_precision, alpha=0.12, color="#16A34A")

    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#3B82F6"]
    for k, color in zip(K_VALUES, colors):
        r_k = results["averaged"][k]["recall"]
        p_k = results["averaged"][k]["precision"]
        ax.scatter(
            r_k,
            p_k,
            color=color,
            s=90,
            zorder=5,
            label=f"HP@{k}={p_k:.3f}  HR@{k}={r_k:.3f}",
        )

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Average Highlight Precision-Recall Curve", fontsize=14)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_query_precision_recall(
    results: dict[str, Any], out_dir: str = PER_QUERY_PLOT_DIR
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for idx, (query, p_curve, r_curve) in enumerate(
        zip(results["queries"], results["p_curves"], results["r_curves"]),
        start=1,
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            r_curve, p_curve, color="#16A34A", linewidth=2, label="Highlight P-R curve"
        )
        ax.fill_between(r_curve, p_curve, alpha=0.10, color="#16A34A")

        query_metrics = results["per_query"][query]
        colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#3B82F6"]
        for k, color in zip(K_VALUES, colors):
            ax.scatter(
                query_metrics["recall"][k],
                query_metrics["precision"][k],
                color=color,
                s=70,
                zorder=5,
                label=f"@{k}",
            )

        title = query if len(query) <= 55 else query[:52] + "..."
        ax.set_title(f"Highlight P-R: {title}", fontsize=10)
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right", title="Cut-off K")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        safe_name = re.sub(r"[^\w]+", "_", query)[:60]
        output_path = os.path.join(out_dir, f"{idx:02d}_{safe_name}.png")
        plt.savefig(output_path, dpi=120)
        plt.close()


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  LLM HIGHLIGHT EVALUATION SUMMARY")
    print("  Positive prediction = result has >=1 LLM quote")
    print("  Positive gold label = result has >=1 annotated highlight phrase")
    print("  TP = predicted quote matches at least one gold phrase")
    print("=" * 72)
    print(f"\n{'K':>6} {'H-Precision':>14} {'H-Recall':>12} {'H-F1':>10}")
    print("-" * 50)
    for k in K_VALUES:
        p = results["averaged"][k]["precision"]
        r = results["averaged"][k]["recall"]
        f = results["averaged"][k]["f1"]
        print(f"{k:>6} {p:>14.4f} {r:>12.4f} {f:>10.4f}")


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated text highlighting quality."
    )
    parser.add_argument("--query-results", default=QUERY_RESULTS_FILE)
    parser.add_argument("--annotations", default=HIGHLIGHT_ANNOTATED_FILE)
    parser.add_argument("--predictions", default=PREDICTIONS_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Only process first N queries when generating predictions (debug/cost control).",
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse cached predictions file if it exists (recommended).",
    )
    args = parser.parse_args()

    _ensure_results_dir()

    print(f"Parsing query results from '{args.query_results}' ...")
    query_results = parse_query_results(args.query_results)
    print(f"  Loaded {len(query_results)} queries")

    print(f"Parsing highlight annotations from '{args.annotations}' ...")
    annotations = parse_annotated_highlights(
        args.annotations, known_queries=set(query_results.keys())
    )
    print(f"  Loaded {len(annotations)} queries")

    predictions_path = Path(args.predictions)
    if args.reuse_predictions and predictions_path.exists():
        print(f"Loading cached predictions from '{args.predictions}' ...")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    else:
        print("Generating LLM highlight predictions ...")
        predictions = generate_predictions(
            query_results=query_results,
            model_name=args.model,
            batch_size=args.batch_size,
            max_queries=args.max_queries,
        )
        predictions_path.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  Saved predictions to '{args.predictions}'")

    print("Computing highlight metrics ...")
    results = compute_highlight_metrics(annotations, predictions)

    print_summary(results)

    print("\nSaving highlight evaluation artifacts ...")
    save_metrics_per_query(results, PER_QUERY_METRICS_PATH)
    save_metrics_averaged(results, AVERAGED_METRICS_PATH)
    save_pr_curve_per_query(results, PR_CURVE_PATH)
    plot_avg_precision_recall(results, AVG_PLOT_PATH)
    plot_per_query_precision_recall(results, PER_QUERY_PLOT_DIR)

    print(f"  Per-query metrics  -> '{PER_QUERY_METRICS_PATH}'")
    print(f"  Averaged metrics   -> '{AVERAGED_METRICS_PATH}'")
    print(f"  P-R curve CSV      -> '{PR_CURVE_PATH}'")
    print(f"  Avg P-R plot       -> '{AVG_PLOT_PATH}'")
    print(f"  Per-query plots    -> '{PER_QUERY_PLOT_DIR}/'")


if __name__ == "__main__":
    main()
