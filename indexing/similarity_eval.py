import argparse
import csv
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_OFFLINE"] = "1"

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

MODEL_PATH = "../../../Embed_Models/"


@dataclass
class Sample:
    group_id: int
    topic: str
    variant_id: int
    text: str


@dataclass
class ModelSimilarityResult:
    model_name: str
    avg_intra_group: float          # mean cosine sim between variants in same group
    avg_inter_group: float          # mean cosine sim between variants across groups
    separation: float               # intra − inter  (higher = better discrimination)
    per_group: dict = field(default_factory=dict)  # group_id → mean intra sim
    encode_seconds: float = 0.0


def load_dataset(csv_path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                Sample(
                    group_id=int(row["group_id"]),
                    topic=row["topic"],
                    variant_id=int(row["variant_id"]),
                    text=row["text"],
                )
            )
    return samples


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def pairwise_cosine(vecs: list[np.ndarray]) -> list[float]:
    """All unique pairs among a list of vectors."""
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine_sim(vecs[i], vecs[j]))
    return sims


def sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_model(
    model_name: str,
    model_path: str,
    samples: list[Sample],
    device: str,
) -> ModelSimilarityResult:
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    texts = [s.text for s in samples]
    sync(device)
    t0 = time.perf_counter()
    embeddings: np.ndarray = model.encode(texts, batch_size=4, convert_to_numpy=True, show_progress_bar=False)
    sync(device)
    encode_seconds = time.perf_counter() - t0


    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for sample, emb in zip(samples, embeddings):
        groups[sample.group_id].append(emb)

    intra_all: list[float] = []
    per_group: dict[int, float] = {}
    for gid, vecs in groups.items():
        sims = pairwise_cosine(vecs)
        per_group[gid] = statistics.mean(sims) if sims else 0.0
        intra_all.extend(sims)

    group_ids = sorted(groups.keys())
    inter_all: list[float] = []
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            gi, gj = group_ids[i], group_ids[j]
            for vi in groups[gi]:
                for vj in groups[gj]:
                    inter_all.append(cosine_sim(vi, vj))

    avg_intra = statistics.mean(intra_all) if intra_all else 0.0
    avg_inter = statistics.mean(inter_all) if inter_all else 0.0

    return ModelSimilarityResult(
        model_name=model_name,
        avg_intra_group=avg_intra,
        avg_inter_group=avg_inter,
        separation=avg_intra - avg_inter,
        per_group=per_group,
        encode_seconds=encode_seconds,
    )


def print_summary(results: list[ModelSimilarityResult], samples: list[Sample]) -> None:
    topic_map = {s.group_id: s.topic for s in samples}
    group_ids = sorted(topic_map.keys())

    print("\nSimilarity Evaluation Results")
    print("=" * 110)
    print(
        f"{'Model':55} {'Intra↑':>8} {'Inter↓':>8} {'Sep↑':>8} {'Encode(s)':>10}"
    )
    print("-" * 110)

    for r in sorted(results, key=lambda x: -x.separation):
        print(
            f"{r.model_name:55} "
            f"{r.avg_intra_group:8.4f} "
            f"{r.avg_inter_group:8.4f} "
            f"{r.separation:8.4f} "
            f"{r.encode_seconds:10.2f}"
        )

    print("-" * 110)
    print("\nIntra-group scores per topic (best model first):\n")

    best = sorted(results, key=lambda x: -x.separation)

    header = f"{'Topic':45}" + "".join(f"  {r.model_name.split('/')[-1][:12]:>12}" for r in best[:5])
    print(header)
    print("-" * len(header))
    for gid in group_ids:
        topic = topic_map[gid].split(":")[0][:44]
        row = f"{topic:45}"
        for r in best[:5]:
            row += f"  {r.per_group.get(gid, 0.0):12.4f}"
        print(row)

    print()


def save_csv(results: list[ModelSimilarityResult], output_path: str, samples: list[Sample]) -> None:
    topic_map = {s.group_id: s.topic for s in samples}
    group_ids = sorted(topic_map.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Summary sheet
        writer.writerow(["model", "avg_intra", "avg_inter", "separation", "encode_seconds"])
        for r in sorted(results, key=lambda x: -x.separation):
            writer.writerow([r.model_name, f"{r.avg_intra_group:.6f}", f"{r.avg_inter_group:.6f}",
                             f"{r.separation:.6f}", f"{r.encode_seconds:.3f}"])
        writer.writerow([])
        # Per-group sheet
        header = ["group_id", "topic"] + [r.model_name for r in results]
        writer.writerow(header)
        for gid in group_ids:
            row = [gid, topic_map[gid]] + [f"{r.per_group.get(gid, 0.0):.6f}" for r in results]
            writer.writerow(row)

    print(f"Results saved to: {output_path}")


def resolve_model_path(model_name: str) -> str | None:
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception as e:
        print(f"  Could not resolve {model_name}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate encoder similarity on semantic triplets.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--csv", default="encoder_test_dataset.csv", help="Path to the dataset CSV")
    parser.add_argument("--output", default="similarity_results.csv", help="Path for output CSV")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model names to run (default: all in MODELS list)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    samples = load_dataset(args.csv)
    print(f"Loaded {len(samples)} samples across {len(set(s.group_id for s in samples))} groups.\n")

    model_names = args.models if args.models else MODELS
    results: list[ModelSimilarityResult] = []
    failed = 0

    for model_name in model_names:
        print(f"Evaluating: {model_name}")
        path = resolve_model_path(model_name)
        if path is None:
            failed += 1
            continue
        try:
            result = evaluate_model(model_name, path, samples, device)
            results.append(result)
            print(
                f"  intra={result.avg_intra_group:.4f}  "
                f"inter={result.avg_inter_group:.4f}  "
                f"sep={result.separation:.4f}  "
                f"({result.encode_seconds:.1f}s)"
            )
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")

    if results:
        print_summary(results, samples)
        save_csv(results, args.output, samples)

    print(f"\nDone. {len(results)} models evaluated, {failed} failed.")


if __name__ == "__main__":
    main()
