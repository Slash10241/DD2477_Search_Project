import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
from elasticsearch import Elasticsearch

import evaluate_embeddings_llm_judge as ev


def precision_at_k(relevances: list[int], k: int) -> float:
    vals = relevances[:k]
    if not vals:
        return 0.0
    return sum(1 for r in vals if r > 0) / len(vals)


def reciprocal_rank(relevances: list[int]) -> float:
    for i, r in enumerate(relevances, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(relevances: list[int], k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"


def truncate_text(text: str, max_len: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def load_queries_from_json(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("queries JSON file must be a JSON list of strings")

    return data


def print_summary_table(summary_rows: list[dict], top_k: int) -> None:
    print("\n" + "=" * 110)
    print("EMBEDDING MODEL EVALUATION SUMMARY")
    print("=" * 110)
    header = (
        f"{'Model':52} "
        f"{f'P@{top_k}':>8} "
        f"{'MRR':>8} "
        f"{f'nDCG@{top_k}':>10} "
        f"{'EmbedTime':>12}"
    )
    print(header)
    print("-" * 110)

    for row in sorted(summary_rows, key=lambda x: x[f"nDCG@{top_k}"], reverse=True):
        print(
            f"{row['model'][:52]:52} "
            f"{row[f'P@{top_k}']:.4f} "
            f"{row['MRR']:.4f} "
            f"{row[f'nDCG@{top_k}']:.4f} "
            f"{format_seconds(row['embedding_time_seconds']):>12}"
        )

    print("=" * 110)


def save_json(output_json_path: str, payload: dict) -> None:
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pretty_text(
    output_txt_path: str,
    summary_rows: list[dict],
    all_results: list[dict],
    config: dict,
    top_k: int,
) -> None:
    lines: list[str] = []

    lines.append("EMBEDDING MODEL EVALUATION REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 100)
    for k, v in config.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 100)
    summary_sorted = sorted(summary_rows, key=lambda x: x[f"nDCG@{top_k}"], reverse=True)
    for row in summary_sorted:
        lines.append(f"Model: {row['model']}")
        lines.append(f"  P@{top_k}: {row[f'P@{top_k}']:.4f}")
        lines.append(f"  MRR: {row['MRR']:.4f}")
        lines.append(f"  nDCG@{top_k}: {row[f'nDCG@{top_k}']:.4f}")
        lines.append(f"  Embedding time: {format_seconds(row['embedding_time_seconds'])}")
        lines.append("")

    lines.append("Detailed Retrieval Results")
    lines.append("-" * 100)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for item in all_results:
        key = (item["model_name"], item["query"])
        grouped.setdefault(key, []).append(item)

    for (model_name, query), items in grouped.items():
        items = sorted(items, key=lambda x: x["rank"])
        lines.append(f"Model: {model_name}")
        lines.append(f"Query: {query}")
        lines.append("")

        for item in items:
            lines.append(f"  Rank: {item['rank']}")
            lines.append(f"  Clip ID: {item['clip_id']}")
            lines.append(f"  Retrieval score: {item['retrieval_score']:.6f}")
            lines.append(f"  Final relevance: {item['relevance']}")
            lines.append(f"  Judge average score: {item['judge_average_score']:.3f}")
            lines.append(f"  Judge raw scores: {item['judge_scores']}")
            lines.append(
                f"  Show / Episode: {item['show_filename_prefix']} / {item['episode_filename_prefix']}"
            )
            lines.append(
                f"  Time span: {item['start_time']:.3f}s -> {item['end_time']:.3f}s"
            )
            lines.append(f"  Text preview: {truncate_text(item['text'], 300)}")
            lines.append(f"  Judge reason: {item['judge_reason']}")
            lines.append("")

        lines.append("-" * 100)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_checkpoint(
    output_json_path: str,
    summary_rows: list[dict],
    all_results: list[dict],
    config: dict,
) -> None:
    payload = {
        "config": config,
        "summary": summary_rows,
        "results": all_results,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

        
def evaluate_model(
    es: Elasticsearch,
    model_name: str,
    queries: list[str],
    chunk_docs: list[ev.ChunkDoc],
    top_k: int,
    device: str,
    judge_model: str,
    num_judgments: int,
    temperature: float,
    output_json_path: str,
    summary_rows_so_far: list[dict],
    all_results_so_far: list[dict],
    config: dict,
) -> tuple[dict, list[dict]]:
    print(f"\nEvaluating model: {model_name}")
    print("-" * 100)

    model = ev.load_embedding_model(model_name=model_name, device=device)

    start_embed = time.perf_counter()
    chunk_embeddings = ev.embed_texts_with_model(
        model=model,
        texts=[c.text for c in chunk_docs],
        batch_size=32,
    )
    embedding_time_seconds = time.perf_counter() - start_embed
    print(f"Chunk embeddings computed in {format_seconds(embedding_time_seconds)}")

    per_query_metrics: list[dict] = []
    all_model_results: list[dict] = []

    for query in queries:
        print(f"\nQuery: {query}")

        try:
            retrieved_results = ev.retrieve_top_k_with_loaded_model(
                query=query,
                model=model,
                chunk_docs=chunk_docs,
                chunk_embeddings=chunk_embeddings,
                top_k=top_k,
            )

            judged_results = ev.judge_retrieved_results(
                query=query,
                model_name=model_name,
                retrieved_results=retrieved_results,
                judge_model=judge_model,
                num_judgments=num_judgments,
                temperature=temperature,
            )

            relevances = [r.relevance for r in sorted(judged_results, key=lambda x: x.rank)]
            query_metrics = {
                "query": query,
                f"P@{top_k}": precision_at_k(relevances, top_k),
                "MRR": reciprocal_rank(relevances),
                f"nDCG@{top_k}": ndcg_at_k(relevances, top_k),
            }
            per_query_metrics.append(query_metrics)

            print(
                f"  P@{top_k}={query_metrics[f'P@{top_k}']:.4f}, "
                f"MRR={query_metrics['MRR']:.4f}, "
                f"nDCG@{top_k}={query_metrics[f'nDCG@{top_k}']:.4f}"
            )

            for jr in judged_results:
                all_model_results.append(
                    {
                        "query": jr.query,
                        "model_name": jr.model_name,
                        "clip_id": jr.clip_id,
                        "rank": jr.rank,
                        "retrieval_score": jr.retrieval_score,
                        "relevance": jr.relevance,
                        "judge_reason": jr.judge_reason,
                        "judge_scores": jr.judge_scores,
                        "judge_reasons": jr.judge_reasons,
                        "judge_average_score": jr.judge_average_score,
                        "episode_filename_prefix": jr.episode_filename_prefix,
                        "show_filename_prefix": jr.show_filename_prefix,
                        "start_time": jr.start_time,
                        "end_time": jr.end_time,
                        "text": jr.text,
                    }
                )

            # save partial progress after every successful query
            partial_summary = {
                "model": model_name,
                f"P@{top_k}": float(np.mean([m[f"P@{top_k}"] for m in per_query_metrics])),
                "MRR": float(np.mean([m["MRR"] for m in per_query_metrics])),
                f"nDCG@{top_k}": float(np.mean([m[f"nDCG@{top_k}"] for m in per_query_metrics])),
                "embedding_time_seconds": embedding_time_seconds,
                "completed_queries": len(per_query_metrics),
                "total_queries": len(queries),
            }

            current_summary_rows = summary_rows_so_far + [partial_summary]
            current_all_results = all_results_so_far + all_model_results

            save_checkpoint(
                output_json_path=output_json_path,
                summary_rows=current_summary_rows,
                all_results=current_all_results,
                config=config,
            )

        except Exception as e:
            print(f"  Failed on query '{query}': {e}")

            partial_summary = {
                "model": model_name,
                f"P@{top_k}": float(np.mean([m[f"P@{top_k}"] for m in per_query_metrics])) if per_query_metrics else 0.0,
                "MRR": float(np.mean([m["MRR"] for m in per_query_metrics])) if per_query_metrics else 0.0,
                f"nDCG@{top_k}": float(np.mean([m[f"nDCG@{top_k}"] for m in per_query_metrics])) if per_query_metrics else 0.0,
                "embedding_time_seconds": embedding_time_seconds,
                "completed_queries": len(per_query_metrics),
                "total_queries": len(queries),
                "last_error": str(e),
            }

            current_summary_rows = summary_rows_so_far + [partial_summary]
            current_all_results = all_results_so_far + all_model_results

            save_checkpoint(
                output_json_path=output_json_path,
                summary_rows=current_summary_rows,
                all_results=current_all_results,
                config=config,
            )

            # continue instead of crashing whole run
            continue

    summary = {
        "model": model_name,
        f"P@{top_k}": float(np.mean([m[f"P@{top_k}"] for m in per_query_metrics])) if per_query_metrics else 0.0,
        "MRR": float(np.mean([m["MRR"] for m in per_query_metrics])) if per_query_metrics else 0.0,
        f"nDCG@{top_k}": float(np.mean([m[f"nDCG@{top_k}"] for m in per_query_metrics])) if per_query_metrics else 0.0,
        "embedding_time_seconds": embedding_time_seconds,
    }

    print(
        f"\nModel summary -> "
        f"P@{top_k}={summary[f'P@{top_k}']:.4f}, "
        f"MRR={summary['MRR']:.4f}, "
        f"nDCG@{top_k}={summary[f'nDCG@{top_k}']:.4f}, "
        f"EmbedTime={format_seconds(summary['embedding_time_seconds'])}"
    )

    return summary, all_model_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models with Ollama LLM judging."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Elasticsearch API key",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for embedding model inference",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=3000,
        help="Limit number of chunks loaded from Elasticsearch",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved results per query",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-oss:120b-cloud",
        help="Ollama judge model name",
    )
    parser.add_argument(
        "--num-judgments",
        type=int,
        default=3,
        help="How many judge calls per (query, chunk) pair",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Judge temperature",
    )
    parser.add_argument(
        "--queries-json",
        type=str,
        default=None,
        help="Optional JSON file containing a list of queries",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of embedding model names to evaluate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="embedding_judge_results.json",
        help="Path to save full JSON results",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="embedding_judge_results.txt",
        help="Path to save pretty text report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries = (
        load_queries_from_json(args.queries_json)
        if args.queries_json is not None
        else ev.DEFAULT_QUERIES
    )

    models_to_test = args.models if args.models else ev.MODELS

    es = Elasticsearch(ev.ES_HOST, api_key=args.api_key)

    print(f"Loading chunks from Elasticsearch index '{ev.INDEX_NAME}' ...")
    chunk_docs = ev.load_chunks_from_es(
        es=es,
        index_name=ev.INDEX_NAME,
        limit=args.chunk_limit,
    )
    print(f"Loaded {len(chunk_docs)} chunks")

    summary_rows: list[dict] = []
    all_results: list[dict] = []

    for model_name in models_to_test:
        summary, model_results = evaluate_model(
            es=es,
            model_name=model_name,
            queries=queries,
            chunk_docs=chunk_docs,
            top_k=args.top_k,
            device=args.device,
            judge_model=args.judge_model,
            num_judgments=args.num_judgments,
            temperature=args.temperature,
            output_json_path=args.output_json,
            summary_rows_so_far=summary_rows,
            all_results_so_far=all_results,
            config={
                "es_host": ev.ES_HOST,
                "index_name": ev.INDEX_NAME,
                "chunk_limit": args.chunk_limit,
                "top_k": args.top_k,
                "device": args.device,
                "judge_model": args.judge_model,
                "num_judgments": args.num_judgments,
                "temperature": args.temperature,
                "queries": queries,
                "models": models_to_test,
            },
        )
        summary_rows.append(summary)
        all_results.extend(model_results)

    print_summary_table(summary_rows, args.top_k)

    payload = {
        "config": {
            "es_host": ev.ES_HOST,
            "index_name": ev.INDEX_NAME,
            "chunk_limit": args.chunk_limit,
            "top_k": args.top_k,
            "device": args.device,
            "judge_model": args.judge_model,
            "num_judgments": args.num_judgments,
            "temperature": args.temperature,
            "queries": queries,
            "models": models_to_test,
        },
        "summary": summary_rows,
        "results": all_results,
    }

    save_json(args.output_json, payload)
    save_pretty_text(
        output_txt_path=args.output_txt,
        summary_rows=summary_rows,
        all_results=all_results,
        config=payload["config"],
        top_k=args.top_k,
    )

    print(f"\nSaved JSON results to: {args.output_json}")
    print(f"Saved text report to: {args.output_txt}")


if __name__ == "__main__":
    main()