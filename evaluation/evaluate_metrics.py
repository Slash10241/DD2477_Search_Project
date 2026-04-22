"""
Evaluation script for Gemini-rated podcast search results.

Parses query_results_rel_{mode}.txt files for hybrid, lexical, and vector search
and computes:
  - Precision@K and Recall@K for K in {5, 10, 15, 20}
  - Average Precision–Recall curve across all queries
  - DCG@K and nDCG@K for K in {5, 10, 15, 20}

Relevance threshold: a result is considered "relevant" if Gemini score >= 2.
For nDCG, the ideal ranking is computed from the UNION of results across all three
search methods per query (deduped by show+episode, top-20 by relevance).  This
gives a single shared IDCG denominator so nDCG scores are directly comparable
across methods rather than each method being normalised against its own best-case.

Outputs per search type (in results_hybrid/, results_lexical/, results_vector/):
  - metrics_per_query.csv        : P@K, R@K, DCG@K, nDCG@K for every query
  - metrics_averaged.csv         : macro-averaged metrics across all queries
  - pr_curve_per_query.csv       : full precision-recall curve (rank 1-N) per query
  - pr_curve.png                 : averaged interpolated P-R curve
  - pr_curves_per_query/         : one PNG per query showing its P-R curve

Additionally outputs:
  - results_comparison/metrics_comparison.csv  : side-by-side averaged metrics for all three
  - results_comparison/comparison_plot.png     : bar chart comparing nDCG@K across search types
  - results_comparison/combined_pr_curve.png   : combined avg P-R curve for all three modes
"""

import csv
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

SEARCH_TYPES = {
    "hybrid":  "query_outputs/query_results_hybrid.txt",
    "lexical": "query_outputs/query_results_lexical.txt",
    "vector":  "query_outputs/query_results_vector.txt",
}

RELEVANCE_THRESHOLD = 2
K_VALUES = [5, 10, 15, 20]          # max results per query is 20
PER_QUERY_PLOT_DIR  = "pr_curves_per_query"
COMPARISON_DIR      = "results_comparison"

def results_folder(search_type: str) -> str:
    return f"results_{search_type}"


# ── 1. Parser ─────────────────────────────────────────────────────────────────

def parse_results_file(filepath: str) -> dict[str, list[dict]]:
    """
    Parses the new query_results_rel_{mode}.txt format produced by getRankingswithRel.py.

    Expected block structure (repeated per query):
        QUERY: <query text>
        RESULTS FOUND: <N>
        ----------------------------------------
        [1]
        Relevance: X/3
        Podcast  : <show name>
        Episode  : <episode name>
        Content  : <text>
        Time     : HH:MM:SS – HH:MM:SS

        [2]
        ...
        ============================================================

    Returns:
        {
            "query text": [
                {"show": str, "episode": str, "relevance": int},
                ...   # up to MAX_RESULTS results, in ranked order
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

        # ── New query block ───────────────────────────────────────────────────
        if line.startswith("QUERY:"):
            # Save previous query
            if current_query is not None:
                data[current_query] = current_results
            current_query   = line[len("QUERY:"):].strip()
            current_results = []
            i += 1
            continue

        # ── Result entry header: [rank] ───────────────────────────────────────
        if re.match(r'^\[\d+\]$', line):
            # Parse the following fields
            result = {}
            i += 1
            while i < len(lines):
                inner = lines[i].strip()

                if inner.startswith("Relevance:"):
                    # "Relevance: X/3"  →  extract X
                    m = re.search(r'(\d+)/3', inner)
                    result["relevance"] = int(m.group(1)) if m else 0

                elif inner.startswith("Podcast  :") or inner.startswith("Podcast:"):
                    result["show"] = re.sub(r'^Podcast\s*:\s*', '', inner).strip()

                elif inner.startswith("Episode  :") or inner.startswith("Episode:"):
                    result["episode"] = re.sub(r'^Episode\s*:\s*', '', inner).strip()

                elif inner.startswith("Content  :") or inner.startswith("Content:"):
                    # content is not needed for metrics, skip
                    pass

                elif inner.startswith("Time     :") or inner.startswith("Time:"):
                    # time range, not needed for metrics
                    pass

                elif re.match(r'^\[\d+\]$', inner) or inner.startswith("QUERY:") \
                        or inner.startswith("="):
                    # Next result or next query — don't advance i here
                    break

                i += 1

            # Only append if we got at minimum a relevance score
            if "relevance" in result:
                result.setdefault("show",    "")
                result.setdefault("episode", "")
                current_results.append(result)
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
    total_relevant = sum(1 for r in results if is_relevant(r["relevance"]))
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for r in results[:k] if is_relevant(r["relevance"]))
    return retrieved_relevant / total_relevant


def compute_precision_recall(data: dict) -> dict:
    per_query            = {}
    avg_p                = {k: [] for k in K_VALUES}
    avg_r                = {k: [] for k in K_VALUES}
    all_precision_curves = []
    all_recall_curves    = []

    for query, results in data.items():
        n              = len(results)
        total_relevant = sum(1 for r in results if is_relevant(r["relevance"]))

        p_at_k = {k: precision_at_k(results, k) for k in K_VALUES}
        r_at_k = {k: recall_at_k(results, k)    for k in K_VALUES}
        per_query[query] = {"precision": p_at_k, "recall": r_at_k}

        for k in K_VALUES:
            avg_p[k].append(p_at_k[k])
            avg_r[k].append(r_at_k[k])

        p_curve, r_curve = [], []
        for k in range(1, n + 1):
            p_curve.append(precision_at_k(results, k))
            r_curve.append(recall_at_k(results, k) if total_relevant > 0 else 0.0)
        all_precision_curves.append(p_curve)
        all_recall_curves.append(r_curve)

    averaged = {
        k: {
            "precision": float(np.mean(avg_p[k])),
            "recall":    float(np.mean(avg_r[k])),
        }
        for k in K_VALUES
    }

    return {
        "per_query": per_query,
        "averaged":  averaged,
        "p_curves":  all_precision_curves,
        "r_curves":  all_recall_curves,
    }


# ── 3. Graded metrics: DCG & nDCG @K ──────────────────────────────────────────

def dcg_at_k(results: list[dict], k: int) -> float:
    score = 0.0
    for i, r in enumerate(results[:k], start=1):
        score += r["relevance"] / math.log2(i + 1)
    return score


def ideal_dcg_at_k(ideal_results: list[dict], k: int) -> float:
    """
    Compute IDCG from a pre-sorted ideal list (shared across all search methods).
    `ideal_results` must already be sorted by relevance descending.
    """
    return dcg_at_k(ideal_results, k)


def ndcg_at_k(results: list[dict], ideal_results: list[dict], k: int) -> float:
    """
    nDCG normalised against the shared ideal pool, not the method's own results.
    """
    idcg = ideal_dcg_at_k(ideal_results, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(results, k) / idcg


# ── 3a. Build shared ideal pool ───────────────────────────────────────────────

def build_ideal_pool(all_data: dict[str, dict]) -> dict[str, list[dict]]:
    """
    For every query, merge results from all three search methods into one pool,
    deduplicate by (show, episode) keeping the highest relevance score seen,
    sort descending by relevance, and cap at MAX_RESULTS (20).

    This pool is used as the IDCG denominator for all three methods so their
    nDCG scores are normalised against the same ideal — enabling fair comparison.

    Args:
        all_data: { search_type: { query: [result_dicts] } }

    Returns:
        { query: [result_dicts sorted by relevance desc, max 20 items] }
    """
    MAX_POOL = max(K_VALUES)  # cap at 20 — matches max retrievable results
    ideal_pool: dict[str, list[dict]] = {}

    # Collect every query seen across all methods
    all_queries: set[str] = set()
    for data in all_data.values():
        all_queries.update(data.keys())

    for query in all_queries:
        # key → best relevance score seen for that (show, episode) pair
        best: dict[tuple, int] = {}
        for data in all_data.values():
            for result in data.get(query, []):
                key = (result.get("show", ""), result.get("episode", ""))
                best[key] = max(best.get(key, 0), result["relevance"])

        # Sort by relevance desc, take top MAX_POOL
        sorted_pool = sorted(
            [{"show": s, "episode": e, "relevance": rel} for (s, e), rel in best.items()],
            key=lambda r: r["relevance"],
            reverse=True,
        )[:MAX_POOL]

        ideal_pool[query] = sorted_pool

    return ideal_pool


def compute_dcg_ndcg(data: dict, ideal_pool: dict) -> dict:
    """
    Compute DCG and nDCG for each query.
    nDCG is normalised against the shared `ideal_pool`, not the method's own results.
    """
    per_query = {}
    avg_dcg   = {k: [] for k in K_VALUES}
    avg_ndcg  = {k: [] for k in K_VALUES}

    for query, results in data.items():
        ideal = ideal_pool.get(query, results)  # fallback to own results if missing
        d = {k: dcg_at_k(results, k)              for k in K_VALUES}
        n = {k: ndcg_at_k(results, ideal, k)      for k in K_VALUES}
        per_query[query] = {"DCG": d, "nDCG": n}
        for k in K_VALUES:
            avg_dcg[k].append(d[k])
            avg_ndcg[k].append(n[k])

    averaged = {
        k: {
            "DCG":  float(np.mean(avg_dcg[k])),
            "nDCG": float(np.mean(avg_ndcg[k])),
        }
        for k in K_VALUES
    }

    return {"per_query": per_query, "averaged": averaged}


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def _interp_pr(p_curve, r_curve, recall_grid):
    """
    Standard IR interpolation: P_interp(r) = max{ P(r') : r' >= r }
    """
    if not r_curve or max(r_curve) == 0:
        return np.zeros(len(recall_grid))

    interp = np.zeros(len(recall_grid))
    for i, r in enumerate(recall_grid):
        # All precision values at recall >= r
        eligible = [p for p, rc in zip(p_curve, r_curve) if rc >= r]
        interp[i] = max(eligible) if eligible else 0.0
    return interp


def plot_avg_precision_recall(pr_results: dict, search_type: str, output_path: str):
    """Individual averaged P-R curve for a single search type."""
    recall_grid       = np.linspace(0, 1, 101)
    interp_precisions = [
        _interp_pr(p, r, recall_grid)
        for p, r in zip(pr_results["p_curves"], pr_results["r_curves"])
    ]
    mean_precision = np.mean(interp_precisions, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(recall_grid, mean_precision, color="#2563EB", linewidth=2.5,
            label=f"Interpolated Precision-Recall Curve ({len(pr_results['per_query'])} queries)")
    ax.fill_between(recall_grid, mean_precision, alpha=0.12, color="#2563EB")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        f"Interpolated Precision-Recall Curve — {search_type.capitalize()} Search\n"
        f"(Spotify 2020 Podcast Dataset, Gemini-rated relevance)",
        fontsize=14,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Avg P-R curve  → '{output_path}'")


def plot_per_query_precision_recall(data: dict, pr_results: dict, out_dir: str):
    """One PNG per query showing its interpolated P-R curve."""
    os.makedirs(out_dir, exist_ok=True)
    queries = list(data.keys())

    recall_grid_pq = np.linspace(0, 1, 101)

    for idx, (query, (p_curve, r_curve)) in enumerate(
        zip(queries, zip(pr_results["p_curves"], pr_results["r_curves"]))
    ):
        interp_p = _interp_pr(p_curve, r_curve, recall_grid_pq)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(recall_grid_pq, interp_p, color="#2563EB", linewidth=2,
                label="Interpolated P-R Curve")
        ax.fill_between(recall_grid_pq, interp_p, alpha=0.10, color="#2563EB")

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        title = query if len(query) <= 55 else query[:52] + "…"
        ax.set_title(f"Interpolated Precision-Recall Curve: {title}", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        safe  = re.sub(r'[^\w]+', '_', query)[:60]
        fpath = os.path.join(out_dir, f"{idx+1:02d}_{safe}.png")
        plt.savefig(fpath, dpi=120)
        plt.close()

    print(f"    Per-query plots → '{out_dir}/' ({len(queries)} files)")


def plot_combined_precision_recall(all_results: dict, output_path: str):
    """
    Combined averaged P-R curve for all three search types on a single chart.
    Each mode gets its own interpolated P-R curve + shaded area.
    """
    COLORS = {
        "hybrid":  "#2563EB",   # blue
        "lexical": "#16A34A",   # green
        "vector":  "#DC2626",   # red
    }
    recall_grid = np.linspace(0, 1, 101)
    fig, ax     = plt.subplots(figsize=(11, 7))

    for stype, res in all_results.items():
        pr    = res["pr"]
        color = COLORS[stype]

        interp = [
            _interp_pr(p, r, recall_grid)
            for p, r in zip(pr["p_curves"], pr["r_curves"])
        ]
        mean_p = np.mean(interp, axis=0)

        ax.plot(recall_grid, mean_p, color=color, linewidth=2.5,
                label=f"{stype.capitalize()} — Interpolated P-R Curve")
        ax.fill_between(recall_grid, mean_p, alpha=0.08, color=color)

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(
        "Combined Interpolated Precision-Recall Curve\n"
        "Hybrid vs Lexical vs Vector Search  ·  Spotify 2020 Podcast Dataset",
        fontsize=14,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, title="Search type")

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Combined P-R plot  → '{output_path}'")


def plot_comparison(all_results: dict, output_path: str):
    """
    Bar chart comparing avg nDCG@K and avg Precision@K across all three search types.
    """
    labels    = [f"@{k}" for k in K_VALUES]
    x         = np.arange(len(K_VALUES))
    width     = 0.25
    colors    = {"hybrid": "#2563EB", "lexical": "#16A34A", "vector": "#DC2626"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        ["nDCG", "precision"],
        ["Average nDCG@K", "Average Precision@K"],
    ):
        for i, (stype, res) in enumerate(all_results.items()):
            pr_res, dcg_res = res["pr"], res["dcg"]
            if metric == "nDCG":
                values = [dcg_res["averaged"][k]["nDCG"] for k in K_VALUES]
            else:
                values = [pr_res["averaged"][k]["precision"] for k in K_VALUES]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=stype.capitalize(),
                          color=colors[stype], alpha=0.85)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("K", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Search Method Comparison — Spotify 2020 Podcast Dataset", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot    → '{output_path}'")


# ── 5. CSV export ─────────────────────────────────────────────────────────────

def save_metrics_per_query(pr_results: dict, dcg_results: dict, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "K", "precision", "recall", "DCG", "nDCG"])
        for q in pr_results["per_query"]:
            for k in K_VALUES:
                writer.writerow([
                    q, k,
                    round(pr_results["per_query"][q]["precision"][k], 6),
                    round(pr_results["per_query"][q]["recall"][k],    6),
                    round(dcg_results["per_query"][q]["DCG"][k],      6),
                    round(dcg_results["per_query"][q]["nDCG"][k],     6),
                ])
    print(f"    Per-query CSV  → '{output_path}'")


def save_metrics_averaged(pr_results: dict, dcg_results: dict, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "avg_precision", "avg_recall", "avg_DCG", "avg_nDCG"])
        for k in K_VALUES:
            writer.writerow([
                k,
                round(pr_results["averaged"][k]["precision"], 6),
                round(pr_results["averaged"][k]["recall"],    6),
                round(dcg_results["averaged"][k]["DCG"],      6),
                round(dcg_results["averaged"][k]["nDCG"],     6),
            ])
    print(f"    Averaged CSV   → '{output_path}'")


def save_pr_curve_per_query(data: dict, pr_results: dict, output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rank", "precision", "recall"])
        for query, p_curve, r_curve in zip(
            data.keys(), pr_results["p_curves"], pr_results["r_curves"]
        ):
            for rank, (p, r) in enumerate(zip(p_curve, r_curve), start=1):
                writer.writerow([query, rank, round(p, 6), round(r, 6)])
    print(f"    P-R curve CSV  → '{output_path}'")


def save_comparison_csv(all_results: dict, output_path: str):
    """Side-by-side averaged metrics for all three search types."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["search_type", "K",
                         "avg_precision", "avg_recall", "avg_DCG", "avg_nDCG"])
        for stype, res in all_results.items():
            for k in K_VALUES:
                writer.writerow([
                    stype, k,
                    round(res["pr"]["averaged"][k]["precision"], 6),
                    round(res["pr"]["averaged"][k]["recall"],    6),
                    round(res["dcg"]["averaged"][k]["DCG"],      6),
                    round(res["dcg"]["averaged"][k]["nDCG"],     6),
                ])
    print(f"  Comparison CSV     → '{output_path}'")


# ── 6. Summary printer ────────────────────────────────────────────────────────

def print_summary(search_type: str, pr_results: dict, dcg_results: dict):
    n_queries = len(pr_results["per_query"])
    print(f"\n{'=' * 62}")
    print(f"  {search_type.upper()} SEARCH  —  {n_queries} queries")
    print(f"  Relevance threshold: score >= {RELEVANCE_THRESHOLD}  |  Max results: {max(K_VALUES)}")
    print(f"{'=' * 62}")
    print(f"\n{'K':>6} {'Precision':>12} {'Recall':>12} {'DCG':>12} {'nDCG':>12}")
    print("-" * 58)
    for k in K_VALUES:
        p = pr_results["averaged"][k]["precision"]
        r = pr_results["averaged"][k]["recall"]
        d = dcg_results["averaged"][k]["DCG"]
        n = dcg_results["averaged"][k]["nDCG"]
        print(f"{k:>6} {p:>12.4f} {r:>12.4f} {d:>12.4f} {n:>12.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results: dict = {}

    # ── Pass 1: parse every file so we can build the shared ideal pool ────────
    print(f"\n{'─' * 55}")
    print(f"  Pass 1 — Parsing all result files …")
    print(f"{'─' * 55}")

    all_data: dict[str, dict] = {}
    for search_type, rel_file in SEARCH_TYPES.items():
        if not Path(rel_file).exists():
            print(f"  [SKIP] File not found: '{rel_file}'")
            continue
        all_data[search_type] = parse_results_file(rel_file)
        print(f"  {search_type.upper():<8} → {len(all_data[search_type])} queries loaded  ({rel_file})")

    if not all_data:
        print("  No files found. Exiting.")
        raise SystemExit(1)

    # ── Build shared ideal pool (union of all three methods per query) ────────
    print(f"\n{'─' * 55}")
    print(f"  Building shared ideal pool (union of all methods) …")
    print(f"{'─' * 55}")
    ideal_pool = build_ideal_pool(all_data)
    print(f"  Ideal pool built for {len(ideal_pool)} queries "
          f"(deduped by show+episode, top-{max(K_VALUES)} by relevance).\n")

    # ── Pass 2: compute metrics per method, using the shared ideal pool ───────
    print(f"{'─' * 55}")
    print(f"  Pass 2 — Computing metrics per search type …")
    print(f"{'─' * 55}")

    for search_type, data in all_data.items():
        res_dir = results_folder(search_type)
        os.makedirs(res_dir, exist_ok=True)

        print(f"\n{'─' * 55}")
        print(f"  Processing: {search_type.upper()}")
        print(f"{'─' * 55}")

        pr_results  = compute_precision_recall(data)
        dcg_results = compute_dcg_ndcg(data, ideal_pool)   # ← shared ideal pool

        print_summary(search_type, pr_results, dcg_results)

        print(f"\n  Saving outputs to '{res_dir}/' …")
        save_metrics_per_query(
            pr_results, dcg_results,
            os.path.join(res_dir, "metrics_per_query.csv"),
        )
        save_metrics_averaged(
            pr_results, dcg_results,
            os.path.join(res_dir, "metrics_averaged.csv"),
        )
        save_pr_curve_per_query(
            data, pr_results,
            os.path.join(res_dir, "pr_curve_per_query.csv"),
        )
        plot_avg_precision_recall(
            pr_results, search_type,
            output_path=os.path.join(res_dir, "pr_curve.png"),
        )
        plot_per_query_precision_recall(
            data, pr_results,
            out_dir=os.path.join(res_dir, PER_QUERY_PLOT_DIR),
        )

        all_results[search_type] = {"pr": pr_results, "dcg": dcg_results}

    # ── Cross-search comparison ───────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'─' * 55}")
        print(f"  Generating comparison outputs …")
        print(f"{'─' * 55}")
        os.makedirs(COMPARISON_DIR, exist_ok=True)

        save_comparison_csv(
            all_results,
            os.path.join(COMPARISON_DIR, "metrics_comparison.csv"),
        )
        plot_comparison(
            all_results,
            output_path=os.path.join(COMPARISON_DIR, "comparison_plot.png"),
        )
        plot_combined_precision_recall(
            all_results,
            output_path=os.path.join(COMPARISON_DIR, "combined_pr_curve.png"),
        )

        # Print side-by-side nDCG@K summary
        print(f"\n  nDCG normalised against shared union ideal pool")
        print(f"  {'K':>4}  " + "  ".join(f"{s.capitalize():>10}" for s in all_results))
        print("  " + "-" * (6 + 12 * len(all_results)))
        for k in K_VALUES:
            row = f"  {k:>4}  "
            row += "  ".join(
                f"{all_results[s]['dcg']['averaged'][k]['nDCG']:>10.4f}"
                for s in all_results
            )
            print(row)

    print("\nDone.\n")