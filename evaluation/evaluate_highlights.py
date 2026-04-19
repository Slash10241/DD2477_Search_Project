"""
Evaluate LLM-generated text highlighting with manual quality ratings (0-3).

New evaluation flow:
1) Run LLM highlighting over ranked transcript chunks.
2) Manually rate each generated highlight quality from 0 to 3.
3) Compute highlight-quality metrics.

Annotation scale per result:
  0 = bad (irrelevant / misleading / no useful highlight)
  1 = weak (partially relevant, low usefulness)
  2 = good (mostly relevant and useful)
  3 = excellent (highly relevant, captures key evidence)

Required inputs:
  - query_results.txt
  - annotated_highlight_quality.txt

Predictions are cached in results/highlight_predictions.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from google import genai

from evaluate_metrics import K_VALUES


QUERY_RESULTS_FILE = "query_results.txt"
QUALITY_ANNOTATED_FILE = "annotated_highlight_quality.txt"
RESULTS_DIR = "results"

PREDICTIONS_FILE = f"{RESULTS_DIR}/highlight_predictions.json"
ANNOTATION_INPUT_FILE = f"{RESULTS_DIR}/highlight_annotation_input.txt"
PER_QUERY_PLOT_DIR = f"{RESULTS_DIR}/highlight_pr_curves_per_query"
AVG_PLOT_PATH = f"{RESULTS_DIR}/highlight_pr_curve.png"
PER_QUERY_METRICS_PATH = f"{RESULTS_DIR}/highlight_metrics_per_query.csv"
AVERAGED_METRICS_PATH = f"{RESULTS_DIR}/highlight_metrics_averaged.csv"
PR_CURVE_PATH = f"{RESULTS_DIR}/highlight_pr_curve_per_query.csv"

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_SIZE = 5
QUALITY_POSITIVE_THRESHOLD = 2


def parse_query_results(filepath: str) -> dict[str, list[dict[str, Any]]]:
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    data: dict[str, list[dict[str, Any]]] = {}

    i = 0
    while i < len(lines):
        query = lines[i].strip()
        if not query:
            i += 1
            continue

        if i + 1 >= len(lines) or not re.fullmatch(r"\d+", lines[i + 1].strip()):
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


def parse_annotated_highlight_quality(
    filepath: str, known_queries: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """
    Format:
      <query>
      <show>
      <episode>
      <quality_score 0-3>
      <blank line>
    """
    data: dict[str, list[dict[str, Any]]] = {}

    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line not in known_queries:
            i += 1
            continue

        query = line

        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        show = lines[i].strip() if i < len(lines) else ""

        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        episode = lines[i].strip() if i < len(lines) else ""

        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        score_line = lines[i].strip() if i < len(lines) else ""

        if show and episode and re.fullmatch(r"[0-3]", score_line):
            data.setdefault(query, []).append(
                {
                    "show": show,
                    "episode": episode,
                    "quality": int(score_line),
                }
            )
            i += 1
            continue

        i += 1

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
    existing_predictions: dict[str, list[dict[str, Any]]] | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    client = genai.Client(api_key=_api_key())
    predictions: dict[str, list[dict[str, Any]]] = existing_predictions or {}

    query_items = list(query_results.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    def _checkpoint_save() -> None:
        if not checkpoint_path:
            return
        Path(checkpoint_path).write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for q_idx, (query, results) in enumerate(query_items, start=1):
        existing_rows = predictions.get(query, [])
        if len(existing_rows) == len(results) and len(existing_rows) > 0:
            print(f"[{q_idx}/{len(query_items)}] Skipping (already saved): {query}")
            continue

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
                        "text": item["text"],
                        "quotes": quotes,
                        "predicted_positive": bool(quotes),
                    }
                )

        predictions[query] = rows
        _checkpoint_save()

    return predictions


def export_annotation_input(
    query_results: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    output_path: str = ANNOTATION_INPUT_FILE,
) -> None:
    """Export a human-friendly file used to rate highlight quality 0-3."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for query, results in query_results.items():
            pred_rows = predictions.get(query, [])
            n = min(len(results), len(pred_rows))
            for i in range(n):
                row = pred_rows[i]
                quotes = row.get("quotes", [])
                quote_str = " || ".join(quotes) if quotes else "NONE"

                fh.write(f"Query: {query}\n")
                fh.write(f"Show: {row.get('show', '')}\n")
                fh.write(f"Episode: {row.get('episode', '')}\n")
                fh.write(f"Predicted highlights: {quote_str}\n")
                fh.write("Transcript:\n")
                fh.write(f"{row.get('text', '')}\n")
                fh.write("-" * 80 + "\n\n")


def is_high_quality(score: int) -> bool:
    return score >= QUALITY_POSITIVE_THRESHOLD


def precision_at_k(pred_pos: list[bool], quality_scores: list[int], k: int) -> float:
    k = min(k, len(pred_pos), len(quality_scores))
    if k <= 0:
        return 0.0

    tp = sum(
        1 for p, s in zip(pred_pos[:k], quality_scores[:k]) if p and is_high_quality(s)
    )
    pp = sum(1 for p in pred_pos[:k] if p)
    if pp == 0:
        return 0.0
    return tp / pp


def recall_at_k(pred_pos: list[bool], quality_scores: list[int], k: int) -> float:
    k = min(k, len(pred_pos), len(quality_scores))
    if k <= 0:
        return 0.0

    tp = sum(
        1 for p, s in zip(pred_pos[:k], quality_scores[:k]) if p and is_high_quality(s)
    )
    total_high_quality = sum(1 for s in quality_scores if is_high_quality(s))
    if total_high_quality == 0:
        return 0.0
    return tp / total_high_quality


def f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def avg_quality_at_k(scores: list[int], k: int) -> float:
    top_k = scores[:k]
    if not top_k:
        return 0.0
    return float(sum(top_k) / len(top_k))


def dcg_at_k(scores: list[int], k: int) -> float:
    total = 0.0
    for i, s in enumerate(scores[:k], start=1):
        total += s / math.log2(i + 1)
    return total


def ndcg_at_k(scores: list[int], k: int) -> float:
    idcg = dcg_at_k(sorted(scores, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(scores, k) / idcg


def compute_highlight_metrics(
    annotations: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_query: dict[str, Any] = {}

    avg_p = {k: [] for k in K_VALUES}
    avg_r = {k: [] for k in K_VALUES}
    avg_f1 = {k: [] for k in K_VALUES}
    avg_q = {k: [] for k in K_VALUES}
    avg_dcg = {k: [] for k in K_VALUES}
    avg_ndcg = {k: [] for k in K_VALUES}

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

        pred_pos = [bool(r.get("predicted_positive", False)) for r in pred_rows[:n]]
        quality_scores = [int(r.get("quality", 0)) for r in ann[:n]]

        p_at_k = {k: precision_at_k(pred_pos, quality_scores, k) for k in K_VALUES}
        r_at_k = {k: recall_at_k(pred_pos, quality_scores, k) for k in K_VALUES}
        f1_at_k = {k: f1_score(p_at_k[k], r_at_k[k]) for k in K_VALUES}
        q_at_k = {k: avg_quality_at_k(quality_scores, k) for k in K_VALUES}
        d_at_k = {k: dcg_at_k(quality_scores, k) for k in K_VALUES}
        n_at_k = {k: ndcg_at_k(quality_scores, k) for k in K_VALUES}

        per_query[query] = {
            "precision": p_at_k,
            "recall": r_at_k,
            "f1": f1_at_k,
            "avg_quality": q_at_k,
            "DCG": d_at_k,
            "nDCG": n_at_k,
            "n_evaluated": n,
        }

        for k in K_VALUES:
            avg_p[k].append(p_at_k[k])
            avg_r[k].append(r_at_k[k])
            avg_f1[k].append(f1_at_k[k])
            avg_q[k].append(q_at_k[k])
            avg_dcg[k].append(d_at_k[k])
            avg_ndcg[k].append(n_at_k[k])

        p_curve: list[float] = []
        r_curve: list[float] = []
        for k in range(1, n + 1):
            p_curve.append(precision_at_k(pred_pos, quality_scores, k))
            r_curve.append(recall_at_k(pred_pos, quality_scores, k))
        p_curves.append(p_curve)
        r_curves.append(r_curve)

    averaged = {
        k: {
            "precision": float(np.mean(avg_p[k])) if avg_p[k] else 0.0,
            "recall": float(np.mean(avg_r[k])) if avg_r[k] else 0.0,
            "f1": float(np.mean(avg_f1[k])) if avg_f1[k] else 0.0,
            "avg_quality": float(np.mean(avg_q[k])) if avg_q[k] else 0.0,
            "DCG": float(np.mean(avg_dcg[k])) if avg_dcg[k] else 0.0,
            "nDCG": float(np.mean(avg_ndcg[k])) if avg_ndcg[k] else 0.0,
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
                "avg_quality",
                "DCG",
                "nDCG",
                "n_evaluated",
            ]
        )
        for query in results["queries"]:
            n_eval = results["per_query"][query]["n_evaluated"]
            for k in K_VALUES:
                q = results["per_query"][query]
                writer.writerow(
                    [
                        query,
                        k,
                        round(q["precision"][k], 6),
                        round(q["recall"][k], 6),
                        round(q["f1"][k], 6),
                        round(q["avg_quality"][k], 6),
                        round(q["DCG"][k], 6),
                        round(q["nDCG"][k], 6),
                        n_eval,
                    ]
                )


def save_metrics_averaged(
    results: dict[str, Any], output_path: str = AVERAGED_METRICS_PATH
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "K",
                "avg_highlight_precision",
                "avg_highlight_recall",
                "avg_highlight_f1",
                "avg_quality",
                "avg_DCG",
                "avg_nDCG",
            ]
        )
        for k in K_VALUES:
            q = results["averaged"][k]
            writer.writerow(
                [
                    k,
                    round(q["precision"], 6),
                    round(q["recall"], 6),
                    round(q["f1"], 6),
                    round(q["avg_quality"], 6),
                    round(q["DCG"], 6),
                    round(q["nDCG"], 6),
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
        label="Mean highlight P-R curve (quality>=2)",
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
        ax.plot(r_curve, p_curve, color="#16A34A", linewidth=2, label="P-R curve")
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
    print("\n" + "=" * 78)
    print("  LLM HIGHLIGHT QUALITY EVALUATION SUMMARY")
    print("  Quality score: 0-3 (higher is better)")
    print(f"  High quality threshold for P/R: >= {QUALITY_POSITIVE_THRESHOLD}")
    print("=" * 78)
    print(
        f"\n{'K':>6} {'H-Precision':>14} {'H-Recall':>12} {'H-F1':>10} {'AvgQ':>10} {'nDCG':>10}"
    )
    print("-" * 74)
    for k in K_VALUES:
        q = results["averaged"][k]
        print(
            f"{k:>6} {q['precision']:>14.4f} {q['recall']:>12.4f} {q['f1']:>10.4f} {q['avg_quality']:>10.4f} {q['nDCG']:>10.4f}"
        )


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated text highlighting with quality ratings (0-3)."
    )
    parser.add_argument("--query-results", default=QUERY_RESULTS_FILE)
    parser.add_argument("--annotations", default=QUALITY_ANNOTATED_FILE)
    parser.add_argument("--predictions", default=PREDICTIONS_FILE)
    parser.add_argument("--annotation-input", default=ANNOTATION_INPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Only process first N queries when generating predictions.",
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse cached predictions file if it exists.",
    )
    args = parser.parse_args()

    _ensure_results_dir()

    print(f"Parsing query results from '{args.query_results}' ...")
    query_results = parse_query_results(args.query_results)
    print(f"  Loaded {len(query_results)} queries")

    predictions_path = Path(args.predictions)
    if args.reuse_predictions and predictions_path.exists():
        print(f"Loading cached predictions from '{args.predictions}' ...")
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    else:
        existing_predictions: dict[str, list[dict[str, Any]]] = {}
        if predictions_path.exists():
            print(
                f"Found existing predictions at '{args.predictions}', resuming generation ..."
            )
            existing_predictions = json.loads(
                predictions_path.read_text(encoding="utf-8")
            )

        print("Generating LLM highlight predictions ...")
        predictions = generate_predictions(
            query_results=query_results,
            model_name=args.model,
            batch_size=args.batch_size,
            max_queries=args.max_queries,
            existing_predictions=existing_predictions,
            checkpoint_path=args.predictions,
        )
        print(f"  Saved predictions to '{args.predictions}'")

    export_annotation_input(
        query_results=query_results,
        predictions=predictions,
        output_path=args.annotation_input,
    )
    print(f"  Exported annotation input to '{args.annotation_input}'")

    print(f"Parsing quality annotations from '{args.annotations}' ...")
    annotations = parse_annotated_highlight_quality(
        args.annotations, known_queries=set(query_results.keys())
    )
    print(f"  Loaded {len(annotations)} queries")

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
