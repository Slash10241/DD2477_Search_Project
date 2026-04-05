"""
asymmetric_retrieval_eval.py
-----------------------------
Tests models in their actual deployment scenario:
  - PASSAGE vectors: encoded from 2-min paragraph chunks (with passage prefix if any)
  - QUERY vectors:   encoded from short 3-5 word queries   (with query prefix if any)

For each group we generate synthetic short queries, then measure:
  - Hit@1:    correct passage is the top-1 result for its query
  - Hit@3:    correct passage is in top-3
  - MRR:      mean reciprocal rank across all queries
  - Avg correct sim:   cosine sim between query and its correct passage
  - Avg distractor sim: cosine sim between query and wrong-group passages
  - Separation: correct_sim - distractor_sim

Compare to symmetric scores (same model, no prefixes, passage-to-passage)
to see how much asymmetric design helps or hurts.

Usage:
    python asymmetric_retrieval_eval.py [--device auto|cpu|cuda]
                                        [--passages encoder_test_dataset.csv]
                                        [--output asymmetric_results.csv]
"""

import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_OFFLINE"] = "1"

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "cointegrated/rubert-tiny2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-small",
    "microsoft/harrier-oss-v1-270m",
    "mixedbread-ai/mxbai-embed-large-v1",
]

# Separate prefixes for query vs passage.
# Models not listed here get empty string for both.
ASYMMETRIC_CONFIG: dict[str, dict] = {
    "BAAI/bge-small-en-v1.5": {
        "query":   "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "BAAI/bge-large-en-v1.5": {
        "query":   "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "BAAI/bge-m3": {
        "query":   "",
        "passage": "",
    },
    "intfloat/multilingual-e5-small": {
        "query":   "query: ",
        "passage": "passage: ",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "query":   "search_query: ",
        "passage": "search_document: ",
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "query":   "Represent this sentence for searching relevant passages: ",
        "passage": "",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "query":   "query: ",
        "passage": "",
    },
}

# ── Synthetic queries (3-5 words) per group ───────────────────────────────────
# Each list entry maps to one passage variant in that group.
# We use 2 queries per group so we get more signal.
QUERIES: dict[int, list[str]] = {
    1:  ["MCU podcast doctor strange",
         "comic book first purchase"],
    2:  ["quitting corporate job creative",
         "leaving stable career burnout"],
    3:  ["solo travel first time",
         "navigating foreign city alone"],
    4:  ["learning guitar as adult",
         "beginner instrument muscle memory"],
    5:  ["mentoring junior colleague work",
         "senior employee helping new hire"],
    6:  ["rediscovering childhood hobby adult",
         "returning to old pastime"],
    7:  ["starting running first race",
         "non-runner training 10K"],
    8:  ["moving abroad expat experience",
         "living in foreign country"],
    9:  ["organising neighborhood community event",
         "first time event planning"],
    10: ["difficult family conversation deferred",
         "honest talk with parent"],
}

# ── Data ──────────────────────────────────────────────────────────────────────
@dataclass
class Passage:
    group_id: int
    topic: str
    variant_id: int
    text: str


@dataclass
class RetrievalResult:
    model_name: str
    # Asymmetric (query prefix / passage prefix)
    asym_hit1: float
    asym_hit3: float
    asym_mrr: float
    asym_correct_sim: float
    asym_distractor_sim: float
    asym_separation: float
    # Symmetric baseline (no prefixes, same model)
    sym_hit1: float
    sym_hit3: float
    sym_mrr: float
    sym_correct_sim: float
    sym_distractor_sim: float
    sym_separation: float
    encode_seconds: float


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_passages(csv_path: str) -> list[Passage]:
    passages = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            passages.append(Passage(
                group_id=int(row["group_id"]),
                topic=row["topic"],
                variant_id=int(row["variant_id"]),
                text=row["text"],
            ))
    return passages


# ── Cosine helpers ────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def rank_passages(query_vec: np.ndarray, passage_vecs: np.ndarray) -> list[int]:
    """Return passage indices sorted by descending cosine similarity to query."""
    norms = np.linalg.norm(passage_vecs, axis=1, keepdims=True) + 1e-10
    normed = passage_vecs / norms
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    sims = normed @ q
    return list(np.argsort(-sims))


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ── Core eval ─────────────────────────────────────────────────────────────────
def run_retrieval(
    model: SentenceTransformer,
    passages: list[Passage],
    query_prefix: str,
    passage_prefix: str,
    device: str,
) -> dict:
    """
    Encode all passages and all queries, then run retrieval.
    Returns dict of aggregated metrics.
    """
    passage_texts = [passage_prefix + p.text for p in passages]
    passage_vecs: np.ndarray = model.encode(
        passage_texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False
    )

    hits1, hits3, rrs = [], [], []
    correct_sims, distractor_sims = [], []

    for group_id, queries in QUERIES.items():
        # Indices of passages belonging to this group (ground truth)
        correct_idxs = {i for i, p in enumerate(passages) if p.group_id == group_id}
        wrong_idxs   = {i for i, p in enumerate(passages) if p.group_id != group_id}

        for query_text in queries:
            q_vec = model.encode(
                query_prefix + query_text, convert_to_numpy=True, show_progress_bar=False
            )

            ranked = rank_passages(q_vec, passage_vecs)

            # Hit metrics
            hit1 = int(ranked[0] in correct_idxs)
            hit3 = int(any(r in correct_idxs for r in ranked[:3]))

            # Reciprocal rank
            rr = 0.0
            for rank, idx in enumerate(ranked, 1):
                if idx in correct_idxs:
                    rr = 1.0 / rank
                    break

            hits1.append(hit1)
            hits3.append(hit3)
            rrs.append(rr)

            # Similarity gap
            c_sims = [cosine_sim(q_vec, passage_vecs[i]) for i in correct_idxs]
            d_sims = [cosine_sim(q_vec, passage_vecs[i]) for i in wrong_idxs]
            correct_sims.append(statistics.mean(c_sims))
            distractor_sims.append(statistics.mean(d_sims))

    avg_correct = statistics.mean(correct_sims)
    avg_distractor = statistics.mean(distractor_sims)

    return {
        "hit1":       statistics.mean(hits1),
        "hit3":       statistics.mean(hits3),
        "mrr":        statistics.mean(rrs),
        "correct_sim":    avg_correct,
        "distractor_sim": avg_distractor,
        "separation": avg_correct - avg_distractor,
    }


def evaluate_model(
    model_name: str,
    model_path: str,
    passages: list[Passage],
    device: str,
) -> RetrievalResult:
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    cfg = ASYMMETRIC_CONFIG.get(model_name, {"query": "", "passage": ""})

    sync(device)
    t0 = time.perf_counter()

    asym = run_retrieval(model, passages, cfg["query"], cfg["passage"], device)
    sym  = run_retrieval(model, passages, "", "", device)

    encode_seconds = time.perf_counter() - t0
    sync(device)

    return RetrievalResult(
        model_name=model_name,
        asym_hit1=asym["hit1"],
        asym_hit3=asym["hit3"],
        asym_mrr=asym["mrr"],
        asym_correct_sim=asym["correct_sim"],
        asym_distractor_sim=asym["distractor_sim"],
        asym_separation=asym["separation"],
        sym_hit1=sym["hit1"],
        sym_hit3=sym["hit3"],
        sym_mrr=sym["mrr"],
        sym_correct_sim=sym["correct_sim"],
        sym_distractor_sim=sym["distractor_sim"],
        sym_separation=sym["separation"],
        encode_seconds=encode_seconds,
    )


# ── Printing ──────────────────────────────────────────────────────────────────
def print_summary(results: list[RetrievalResult]) -> None:
    print("\nAsymmetric Retrieval Evaluation  (query=3-5 words → passage=2-min chunk)")
    print("=" * 120)
    print(f"{'Model':50} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6} {'CorrectSim':>11} {'DistrSim':>9} {'Sep':>7}  {'vs Sym Sep':>10}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: -x.asym_mrr):
        delta = r.asym_separation - r.sym_separation
        delta_str = f"{delta:+.4f}"
        print(
            f"{r.model_name:50} "
            f"{r.asym_hit1:6.3f} "
            f"{r.asym_hit3:6.3f} "
            f"{r.asym_mrr:6.3f} "
            f"{r.asym_correct_sim:11.4f} "
            f"{r.asym_distractor_sim:9.4f} "
            f"{r.asym_separation:7.4f}  "
            f"{delta_str:>10}"
        )
    print("-" * 120)
    print("\nSymmetric baseline (no prefixes, passage→passage for reference):")
    print(f"{'Model':50} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6} {'Sep':>7}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x.sym_mrr):
        print(
            f"{r.model_name:50} "
            f"{r.sym_hit1:6.3f} "
            f"{r.sym_hit3:6.3f} "
            f"{r.sym_mrr:6.3f} "
            f"{r.sym_separation:7.4f}"
        )
    print("-" * 80)


def save_csv(results: list[RetrievalResult], output_path: str) -> None:
    fields = [
        "model",
        "asym_hit1", "asym_hit3", "asym_mrr",
        "asym_correct_sim", "asym_distractor_sim", "asym_separation",
        "sym_hit1", "sym_hit3", "sym_mrr",
        "sym_correct_sim", "sym_distractor_sim", "sym_separation",
        "sep_delta_asym_vs_sym",
        "encode_seconds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(results, key=lambda x: -x.asym_mrr):
            writer.writerow({
                "model": r.model_name,
                "asym_hit1": f"{r.asym_hit1:.4f}",
                "asym_hit3": f"{r.asym_hit3:.4f}",
                "asym_mrr":  f"{r.asym_mrr:.4f}",
                "asym_correct_sim":    f"{r.asym_correct_sim:.4f}",
                "asym_distractor_sim": f"{r.asym_distractor_sim:.4f}",
                "asym_separation":     f"{r.asym_separation:.4f}",
                "sym_hit1": f"{r.sym_hit1:.4f}",
                "sym_hit3": f"{r.sym_hit3:.4f}",
                "sym_mrr":  f"{r.sym_mrr:.4f}",
                "sym_correct_sim":    f"{r.sym_correct_sim:.4f}",
                "sym_distractor_sim": f"{r.sym_distractor_sim:.4f}",
                "sym_separation":     f"{r.sym_separation:.4f}",
                "sep_delta_asym_vs_sym": f"{r.asym_separation - r.sym_separation:+.4f}",
                "encode_seconds": f"{r.encode_seconds:.2f}",
            })
    print(f"\nResults saved to: {output_path}")


def resolve_model_path(model_name: str) -> str | None:
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception as e:
        print(f"  Could not resolve {model_name}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asymmetric retrieval eval: short query → long passage.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--passages", default="encoder_test_dataset.csv",
                        help="Path to the 2-min passage CSV (encoder_test_dataset.csv)")
    parser.add_argument("--output", default="asymmetric_results.csv")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    passages = load_passages(args.passages)
    print(f"Loaded {len(passages)} passages across {len(set(p.group_id for p in passages))} groups.")
    total_queries = sum(len(q) for q in QUERIES.values())
    print(f"Queries: {total_queries} total ({len(QUERIES)} groups × ~2 queries each)\n")

    model_names = args.models if args.models else MODELS
    results: list[RetrievalResult] = []
    failed = 0

    for model_name in model_names:
        cfg = ASYMMETRIC_CONFIG.get(model_name, {"query": "", "passage": ""})
        q_pfx = cfg["query"] or "(none)"
        p_pfx = cfg["passage"] or "(none)"
        print(f"Evaluating: {model_name}")
        print(f"  query prefix='{q_pfx}'  passage prefix='{p_pfx}'")

        path = resolve_model_path(model_name)
        if path is None:
            failed += 1
            continue
        try:
            result = evaluate_model(model_name, path, passages, device)
            results.append(result)
            delta = result.asym_separation - result.sym_separation
            print(
                f"  [ASYM] hit@1={result.asym_hit1:.3f}  mrr={result.asym_mrr:.3f}  sep={result.asym_separation:.4f}"
                f"  [SYM] hit@1={result.sym_hit1:.3f}  mrr={result.sym_mrr:.3f}  sep={result.sym_separation:.4f}"
                f"  delta={delta:+.4f}"
            )
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")

    if results:
        print_summary(results)
        save_csv(results, args.output)

    print(f"\nDone. {len(results)} models evaluated, {failed} failed.")


if __name__ == "__main__":
    main()
