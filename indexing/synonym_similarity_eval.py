import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

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

# Optional per-model input prefixes. Set to "" to disable for a model.
# nomic-embed expects a task prefix; BGE models also have optional query prefixes.
PREFIX_MAP: dict[str, str] = {
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
    "BAAI/bge-large-en-v1.5": "Represent this word for semantic similarity: ",
    "BAAI/bge-small-en-v1.5": "Represent this word for semantic similarity: ",
    "BAAI/bge-m3": "",
    "intfloat/multilingual-e5-small": "query: ",
}


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SynonymEntry:
    group_id: int
    concept: str
    part_of_speech: str
    synonym_id: int
    word: str
    proximity_note: str


@dataclass
class ModelResult:
    model_name: str
    avg_intra_group: float
    avg_inter_group: float
    separation: float
    per_group: dict = field(default_factory=dict)   # group_id → {mean, pairs}
    per_proximity: dict = field(default_factory=dict)  # proximity_note → mean sim to anchor
    encode_seconds: float = 0.0


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> list[SynonymEntry]:
    entries: list[SynonymEntry] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries.append(SynonymEntry(
                group_id=int(row["group_id"]),
                concept=row["concept"],
                part_of_speech=row["part_of_speech"],
                synonym_id=int(row["synonym_id"]),
                word=row["word"],
                proximity_note=row["proximity_note"],
            ))
    return entries


# ── Cosine helpers ────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def pairwise_cosine(vecs: list[np.ndarray]) -> list[float]:
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine_sim(vecs[i], vecs[j]))
    return sims


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# ── Per-model evaluation ──────────────────────────────────────────────────────
def evaluate_model(
    model_name: str,
    model_path: str,
    entries: list[SynonymEntry],
    device: str,
) -> ModelResult:
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    prefix = PREFIX_MAP.get(model_name, "")

    texts = [prefix + e.word for e in entries]

    sync(device)
    t0 = time.perf_counter()
    embeddings: np.ndarray = model.encode(
        texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False
    )
    sync(device)
    encode_seconds = time.perf_counter() - t0

    # Index embeddings by group
    groups: dict[int, list[tuple[SynonymEntry, np.ndarray]]] = defaultdict(list)
    for entry, emb in zip(entries, embeddings):
        groups[entry.group_id].append((entry, emb))

    # Intra-group: all pairs within same concept group
    intra_all: list[float] = []
    per_group: dict[int, dict] = {}
    for gid, pairs in groups.items():
        vecs = [p[1] for p in pairs]
        sims = pairwise_cosine(vecs)
        per_group[gid] = {
            "concept": pairs[0][0].concept,
            "mean": statistics.mean(sims) if sims else 0.0,
            "min": min(sims) if sims else 0.0,
            "max": max(sims) if sims else 0.0,
            "words": [p[0].word for p in pairs],
        }
        intra_all.extend(sims)

    # Inter-group: cross-group pairs
    group_ids = sorted(groups.keys())
    inter_all: list[float] = []
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            gi, gj = group_ids[i], group_ids[j]
            for _, vi in groups[gi]:
                for _, vj in groups[gj]:
                    inter_all.append(cosine_sim(vi, vj))

    # Per-proximity: anchor (synonym_id=1) vs others within group
    proximity_sims: dict[str, list[float]] = defaultdict(list)
    for gid, pairs in groups.items():
        anchor = next((emb for e, emb in pairs if e.synonym_id == 1), None)
        if anchor is None:
            continue
        for entry, emb in pairs:
            if entry.synonym_id == 1:
                continue
            sim = cosine_sim(anchor, emb)
            proximity_sims[entry.proximity_note].append(sim)

    per_proximity = {k: statistics.mean(v) for k, v in proximity_sims.items()}

    avg_intra = statistics.mean(intra_all) if intra_all else 0.0
    avg_inter = statistics.mean(inter_all) if inter_all else 0.0

    return ModelResult(
        model_name=model_name,
        avg_intra_group=avg_intra,
        avg_inter_group=avg_inter,
        separation=avg_intra - avg_inter,
        per_group=per_group,
        per_proximity=per_proximity,
        encode_seconds=encode_seconds,
    )


# ── Printing ──────────────────────────────────────────────────────────────────
def print_summary(results: list[ModelResult]) -> None:
    print("\nSynonym Similarity Evaluation Results")
    print("=" * 105)
    print(f"{'Model':55} {'Intra↑':>8} {'Inter↓':>8} {'Sep↑':>8} {'Encode(s)':>10}")
    print("-" * 105)
    for r in sorted(results, key=lambda x: -x.separation):
        print(
            f"{r.model_name:55} "
            f"{r.avg_intra_group:8.4f} "
            f"{r.avg_inter_group:8.4f} "
            f"{r.separation:8.4f} "
            f"{r.encode_seconds:10.3f}"
        )
    print("-" * 105)

    # Per-group breakdown for top 3 models
    top3 = sorted(results, key=lambda x: -x.separation)[:3]
    print(f"\nPer-concept intra similarity (top 3 models):\n")
    header = f"{'Concept':20}" + "".join(f"  {r.model_name.split('/')[-1][:16]:>16}" for r in top3)
    print(header)
    print("-" * len(header))
    group_ids = sorted(top3[0].per_group.keys())
    for gid in group_ids:
        concept = top3[0].per_group[gid]["concept"]
        row = f"{concept:20}"
        for r in top3:
            row += f"  {r.per_group[gid]['mean']:16.4f}"
        print(row)

    # Proximity analysis
    print(f"\nMean similarity to anchor by proximity type (top 3 models):\n")
    prox_labels = ["near-identical", "stronger intensity", "milder / quieter form",
                   "behaviorally expressed", "formal register", "extreme intensity",
                   "morally-flavoured anger", "literary", "chronic / reflective sadness",
                   "situational / defeated", "lonely-tinged sadness"]
    all_prox = sorted(set(k for r in top3 for k in r.per_proximity.keys()))
    header2 = f"{'Proximity type':35}" + "".join(f"  {r.model_name.split('/')[-1][:14]:>14}" for r in top3)
    print(header2)
    print("-" * len(header2))
    for p in all_prox:
        row = f"{p:35}"
        for r in top3:
            val = r.per_proximity.get(p, float('nan'))
            row += f"  {val:14.4f}" if not (val != val) else f"  {'N/A':>14}"
        print(row)


def save_csv(results: list[ModelResult], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "avg_intra", "avg_inter", "separation", "encode_seconds"])
        for r in sorted(results, key=lambda x: -x.separation):
            writer.writerow([r.model_name, f"{r.avg_intra_group:.6f}",
                             f"{r.avg_inter_group:.6f}", f"{r.separation:.6f}",
                             f"{r.encode_seconds:.3f}"])
        writer.writerow([])
        # Per-group
        all_gids = sorted(next(iter(results)).per_group.keys())
        writer.writerow(["group_id", "concept"] + [r.model_name for r in results])
        for gid in all_gids:
            concept = results[0].per_group[gid]["concept"]
            row = [gid, concept] + [f"{r.per_group[gid]['mean']:.6f}" for r in results]
            writer.writerow(row)
        writer.writerow([])
        # Per-proximity
        all_prox = sorted(set(k for r in results for k in r.per_proximity.keys()))
        writer.writerow(["proximity_note"] + [r.model_name for r in results])
        for p in all_prox:
            row = [p] + [f"{r.per_proximity.get(p, float('nan')):.6f}" for r in results]
            writer.writerow(row)
    print(f"\nResults saved to: {output_path}")


def resolve_model_path(model_name: str) -> str | None:
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception as e:
        print(f"  Could not resolve {model_name}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate encoder similarity on synonym groups.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--csv", default="synonym_test_dataset.csv")
    parser.add_argument("--output", default="synonym_similarity_results.csv")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    entries = load_dataset(args.csv)
    n_groups = len(set(e.group_id for e in entries))
    print(f"Loaded {len(entries)} words across {n_groups} synonym groups.\n")

    model_names = args.models if args.models else MODELS
    results: list[ModelResult] = []
    failed = 0

    for model_name in model_names:
        prefix = PREFIX_MAP.get(model_name, "")
        print(f"Evaluating: {model_name}" + (f"  [prefix: '{prefix}']" if prefix else ""))
        path = resolve_model_path(model_name)
        if path is None:
            failed += 1
            continue
        try:
            result = evaluate_model(model_name, path, entries, device)
            results.append(result)
            print(
                f"  intra={result.avg_intra_group:.4f}  "
                f"inter={result.avg_inter_group:.4f}  "
                f"sep={result.separation:.4f}  "
                f"({result.encode_seconds:.3f}s)"
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
