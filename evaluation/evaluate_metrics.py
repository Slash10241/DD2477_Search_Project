"""
Evaluation script for annotated podcast search results.

Parses annotated_results.txt into a dictionary and computes:
  - Precision@K and Recall@K for K in {10, 20, 30, 40, 50}
  - Average Precision–Recall curve across all queries
  - DCG@K and nDCG@K for K in {10, 20, 30, 40, 50}

Relevance threshold: a result is considered "relevant" if score >= 2.
For nDCG, the ideal ranking is computed from the 50 available results only.
For recall, the total relevant count is also computed from the 50 available results.

Outputs:
  - metrics_per_query.csv        : P@K, R@K, DCG@K, nDCG@K for every query
  - metrics_averaged.csv         : macro-averaged metrics across all queries
  - pr_curve_per_query.csv       : full precision-recall curve (rank 1-50) per query
  - pr_curve.png                 : averaged P-R curve with P@K/R@K markers
  - pr_curves_per_query/         : one PNG per query showing its P-R curve
"""

import csv
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

ANNOTATED_FILE = "annotated_results.txt"  # change path if needed
RELEVANCE_THRESHOLD = 2  # scores >= this are "relevant" for binary metrics
K_VALUES = [10, 20, 30, 40, 50]
PER_QUERY_PLOT_DIR = "pr_curves_per_query"
RES_FOLDER = "results"

# ── 1. Parser ─────────────────────────────────────────────────────────────────


def parse_annotated_file(filepath: str) -> dict[str, list[dict]]:
    """
    Returns:
        {
            "query text": [
                {"show": str, "episode": str, "relevance": int},
                ...   # up to 50 results, in ranked order
            ],
            ...
        }
    """
    data: dict[str, list[dict]] = {}
    current_query: str | None = None
    current_results: list[dict] = []

    lines = Path(filepath).read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect a new query block:
        # A query line is followed by a show name, then episode, then an integer (relevance).
        # We identify it by: non-empty line whose content is NOT a digit and
        # whose next non-empty lines form a (show, episode, score) triple.
        if line and not re.match(r"^\d+$", line):
            # Peek ahead to see if this looks like a query (next valid block has a score)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            show_candidate = lines[j].strip() if j < len(lines) else ""
            j2 = j + 1
            episode_candidate = lines[j2].strip() if j2 < len(lines) else ""
            j3 = j2 + 1
            score_candidate = lines[j3].strip() if j3 < len(lines) else ""

            if (
                re.match(r"^[0-3]$", score_candidate)
                and show_candidate
                and episode_candidate
            ):
                # Save previous query
                if current_query is not None:
                    data[current_query] = current_results

                current_query = line
                current_results = []
                i += 1
                continue

        # Detect a result block: show \n episode \n score
        if line and current_query is not None and not re.match(r"^\d+$", line):
            show = line
            i += 1
            episode = lines[i].strip() if i < len(lines) else ""
            i += 1
            score_str = lines[i].strip() if i < len(lines) else ""
            if re.match(r"^[0-3]$", score_str):
                current_results.append(
                    {
                        "show": show,
                        "episode": episode,
                        "relevance": int(score_str),
                    }
                )
                i += 1
                continue

        i += 1

    # Save last query
    if current_query is not None and current_results:
        data[current_query] = current_results

    return data


# ── 2. Binary metrics: Precision & Recall @K ──────────────────────────────────


def is_relevant(score: int) -> bool:
    return score >= RELEVANCE_THRESHOLD


def precision_at_k(results: list[dict], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if is_relevant(r["relevance"])) / len(top_k)


def recall_at_k(results: list[dict], k: int) -> float:
    """Recall denominator = total relevant in the full 50 results."""
    total_relevant = sum(1 for r in results if is_relevant(r["relevance"]))
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in results[:k] if is_relevant(r["relevance"]))
    return retrieved_relevant / total_relevant


def compute_precision_recall(data: dict) -> dict:
    """
    Returns per-query and averaged P@K and R@K for all K_VALUES,
    plus full precision-recall curves for plotting.
    """
    per_query = {}
    avg_p = {k: [] for k in K_VALUES}
    avg_r = {k: [] for k in K_VALUES}

    # For average P-R curve (at every rank position 1..50)
    all_precision_curves = []
    all_recall_curves = []

    for query, results in data.items():
        n = len(results)
        total_relevant = sum(1 for r in results if is_relevant(r["relevance"]))

        p_at_k = {k: precision_at_k(results, k) for k in K_VALUES}
        r_at_k = {k: recall_at_k(results, k) for k in K_VALUES}
        per_query[query] = {"precision": p_at_k, "recall": r_at_k}

        for k in K_VALUES:
            avg_p[k].append(p_at_k[k])
            avg_r[k].append(r_at_k[k])

        # Full curve at every rank
        p_curve, r_curve = [], []
        for k in range(1, n + 1):
            p_curve.append(precision_at_k(results, k))
            r_curve.append(recall_at_k(results, k) if total_relevant > 0 else 0.0)
        all_precision_curves.append(p_curve)
        all_recall_curves.append(r_curve)

    averaged = {
        k: {
            "precision": float(np.mean(avg_p[k])),
            "recall": float(np.mean(avg_r[k])),
        }
        for k in K_VALUES
    }

    return {
        "per_query": per_query,
        "averaged": averaged,
        "p_curves": all_precision_curves,
        "r_curves": all_recall_curves,
    }


# ── 3. Graded metrics: DCG & nDCG @K ──────────────────────────────────────────


def dcg_at_k(results: list[dict], k: int) -> float:
    """DCG with log base 2. Uses graded relevance 0-3."""
    score = 0.0
    for i, r in enumerate(results[:k], start=1):
        rel = r["relevance"]
        score += rel / math.log2(i + 1)
    return score


def ideal_dcg_at_k(results: list[dict], k: int) -> float:
    """
    IDCG: sort the 50 available results by relevance descending,
    then compute DCG@K on that ideal ordering.
    """
    ideal = sorted(results, key=lambda r: r["relevance"], reverse=True)
    return dcg_at_k(ideal, k)


def ndcg_at_k(results: list[dict], k: int) -> float:
    idcg = ideal_dcg_at_k(results, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(results, k) / idcg


def compute_dcg_ndcg(data: dict) -> dict:
    per_query = {}
    avg_dcg = {k: [] for k in K_VALUES}
    avg_ndcg = {k: [] for k in K_VALUES}

    for query, results in data.items():
        d = {k: dcg_at_k(results, k) for k in K_VALUES}
        n = {k: ndcg_at_k(results, k) for k in K_VALUES}
        per_query[query] = {"DCG": d, "nDCG": n}
        for k in K_VALUES:
            avg_dcg[k].append(d[k])
            avg_ndcg[k].append(n[k])

    averaged = {
        k: {
            "DCG": float(np.mean(avg_dcg[k])),
            "nDCG": float(np.mean(avg_ndcg[k])),
        }
        for k in K_VALUES
    }

    return {"per_query": per_query, "averaged": averaged}


# ── 4. Plots ──────────────────────────────────────────────────────────────────


def _interp_pr(p_curve, r_curve, recall_grid):
    """Interpolate a single P-R curve onto a fixed recall grid."""
    if not r_curve or r_curve[-1] == 0:
        return np.zeros(len(recall_grid))
    paired = sorted(zip(r_curve, p_curve))
    r_sorted = [x[0] for x in paired]
    p_sorted = [x[1] for x in paired]
    return np.interp(recall_grid, r_sorted, p_sorted)


def plot_avg_precision_recall(
    pr_results: dict, output_path: str = RES_FOLDER + "/pr_curve.png"
):
    """Average P-R curve across all queries with P@K / R@K markers."""
    recall_grid = np.linspace(0, 1, 101)
    interp_precisions = [
        _interp_pr(p, r, recall_grid)
        for p, r in zip(pr_results["p_curves"], pr_results["r_curves"])
    ]
    mean_precision = np.mean(interp_precisions, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        recall_grid,
        mean_precision,
        color="#2563EB",
        linewidth=2.5,
        label="Mean P-R curve (all 60 queries)",
    )
    ax.fill_between(recall_grid, mean_precision, alpha=0.12, color="#2563EB")

    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#8B5CF6"]
    for k, color in zip(K_VALUES, colors):
        r_k = pr_results["averaged"][k]["recall"]
        p_k = pr_results["averaged"][k]["precision"]
        ax.scatter(r_k, p_k, color=color, s=90, zorder=5, label=f"k={k}")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    # ax.set_title("Average Precision–Recall Curve\n(60 queries, Spotify 2020 Podcast Dataset)", fontsize=14)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Avg P-R curve → '{output_path}'")


def plot_per_query_precision_recall(data: dict, pr_results: dict, out_dir: str):
    """One P-R curve PNG per query, saved in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    recall_grid = np.linspace(0, 1, 101)
    queries = list(data.keys())

    for idx, (query, (p_curve, r_curve)) in enumerate(
        zip(queries, zip(pr_results["p_curves"], pr_results["r_curves"]))
    ):
        interp = _interp_pr(p_curve, r_curve, recall_grid)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(r_curve, p_curve, color="#2563EB", linewidth=2, label="P-R curve")
        ax.fill_between(r_curve, p_curve, alpha=0.10, color="#2563EB")

        colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#8B5CF6"]
        pq = pr_results["per_query"][query]
        for k, color in zip(K_VALUES, colors):
            ax.scatter(
                pq["recall"][k],
                pq["precision"][k],
                color=color,
                s=70,
                zorder=5,
                label=f"@{k}",
            )

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        # Truncate long query for title
        title = query if len(query) <= 55 else query[:52] + "…"
        ax.set_title(f"P-R Curve: {title}", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right", title="Cut-off K")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        # Safe filename: replace non-alphanumeric with underscore, truncate
        safe = re.sub(r"[^\w]+", "_", query)[:60]
        fpath = os.path.join(out_dir, f"{idx + 1:02d}_{safe}.png")
        plt.savefig(fpath, dpi=120)
        plt.close()

    print(f"  Per-query P-R plots → '{out_dir}/' ({len(queries)} files)")


# ── 5. CSV export ─────────────────────────────────────────────────────────────


def save_metrics_per_query(
    pr_results: dict,
    dcg_results: dict,
    output_path: str = RES_FOLDER + "/metrics_per_query.csv",
):
    """
    One row per (query, K) with columns:
      query | K | precision | recall | DCG | nDCG
    """
    queries = list(pr_results["per_query"].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "K", "precision", "recall", "DCG", "nDCG"])
        for q in queries:
            for k in K_VALUES:
                writer.writerow(
                    [
                        q,
                        k,
                        round(pr_results["per_query"][q]["precision"][k], 6),
                        round(pr_results["per_query"][q]["recall"][k], 6),
                        round(dcg_results["per_query"][q]["DCG"][k], 6),
                        round(dcg_results["per_query"][q]["nDCG"][k], 6),
                    ]
                )
    print(f"  Per-query metrics  → '{output_path}'")


def save_metrics_averaged(
    pr_results: dict,
    dcg_results: dict,
    output_path: str = RES_FOLDER + "/metrics_averaged.csv",
):
    """
    One row per K with macro-averaged metrics.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "avg_precision", "avg_recall", "avg_DCG", "avg_nDCG"])
        for k in K_VALUES:
            writer.writerow(
                [
                    k,
                    round(pr_results["averaged"][k]["precision"], 6),
                    round(pr_results["averaged"][k]["recall"], 6),
                    round(dcg_results["averaged"][k]["DCG"], 6),
                    round(dcg_results["averaged"][k]["nDCG"], 6),
                ]
            )
    print(f"  Averaged metrics   → '{output_path}'")


def save_pr_curve_per_query(
    data: dict,
    pr_results: dict,
    output_path: str = RES_FOLDER + "/pr_curve_per_query.csv",
):
    """
    Full precision-recall curve at every rank position (1-50) per query.
    Columns: query | rank | precision | recall
    """
    queries = list(data.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rank", "precision", "recall"])
        for query, p_curve, r_curve in zip(
            queries, pr_results["p_curves"], pr_results["r_curves"]
        ):
            for rank, (p, r) in enumerate(zip(p_curve, r_curve), start=1):
                writer.writerow([query, rank, round(p, 6), round(r, 6)])
    print(f"  P-R curves (full)  → '{output_path}'")


# ── 6. Pretty print summary ───────────────────────────────────────────────────


def print_summary(pr_results: dict, dcg_results: dict):
    print("\n" + "=" * 62)
    print("  RETRIEVAL EVALUATION SUMMARY")
    print("  Relevance threshold: score >= 2  |  Pool: top-50 per query")
    print("=" * 62)
    print(f"\n{'K':>6} {'Precision':>12} {'Recall':>12} {'DCG':>12} {'nDCG':>12}")
    print("-" * 58)
    for k in K_VALUES:
        p = pr_results["averaged"][k]["precision"]
        r = pr_results["averaged"][k]["recall"]
        d = dcg_results["averaged"][k]["DCG"]
        n = dcg_results["averaged"][k]["nDCG"]
        print(f"{k:>6} {p:>12.4f} {r:>12.4f} {d:>12.4f} {n:>12.4f}")

    print("\nPer-query nDCG@10 (sorted desc):")
    ndcg10 = {
        q: dcg_results["per_query"][q]["nDCG"][10] for q in dcg_results["per_query"]
    }
    for q, v in sorted(ndcg10.items(), key=lambda x: x[1], reverse=True):
        print(f"  {v:.4f}  {q}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Parsing '{ANNOTATED_FILE}' …")
    data = parse_annotated_file(ANNOTATED_FILE)
    print(f"  Loaded {len(data)} queries, each with up to 50 results.\n")

    print("Computing Precision / Recall …")
    pr_results = compute_precision_recall(data)

    print("Computing DCG / nDCG …")
    dcg_results = compute_dcg_ndcg(data)

    print_summary(pr_results, dcg_results)

    print("\nSaving CSVs …")
    save_metrics_per_query(
        pr_results, dcg_results, RES_FOLDER + "/metrics_per_query.csv"
    )
    save_metrics_averaged(pr_results, dcg_results, RES_FOLDER + "/metrics_averaged.csv")
    save_pr_curve_per_query(data, pr_results, RES_FOLDER + "/pr_curve_per_query.csv")

    print("\nPlotting …")
    plot_avg_precision_recall(pr_results, output_path=RES_FOLDER + "/pr_curve.png")
    plot_per_query_precision_recall(
        data, pr_results, out_dir=RES_FOLDER + "/" + PER_QUERY_PLOT_DIR
    )
