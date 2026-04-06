import argparse
import json
import re
from dataclasses import dataclass, field

import numpy as np
from elasticsearch import Elasticsearch, helpers
from ollama import chat
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen3-Embedding-0.6B",
]

DEFAULT_QUERIES = [
    "Higgs Boson",
    "terrorism",
    "what Jesus means to me"
]


@dataclass
class ChunkDoc:
    clip_id: str
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float
    text: str


@dataclass
class RetrievedResult:
    clip_id: str
    score: float
    rank: int
    text: str
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float


@dataclass
class JudgedResult:
    query: str
    model_name: str
    clip_id: str
    rank: int
    retrieval_score: float
    relevance: int  # final rounded label in {0,1,2}
    judge_reason: str
    text: str
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float
    judge_scores: list[int] = field(default_factory=list)
    judge_reasons: list[str] = field(default_factory=list)
    judge_average_score: float = 0.0


def make_clip_id(
    episode_filename_prefix: str,
    show_filename_prefix: str,
    start_time: float,
    end_time: float,
) -> str:
    return f"{show_filename_prefix}::{episode_filename_prefix}::{start_time:.3f}::{end_time:.3f}"


def load_chunks_from_es(
    es: Elasticsearch,
    index_name: str,
    limit: int | None = None,
) -> list[ChunkDoc]:
    chunks: list[ChunkDoc] = []

    for hit in helpers.scan(
        es,
        index=index_name,
        query={"query": {"match_all": {}}},
        _source=[
            "episode_filename_prefix",
            "show_filename_prefix",
            "start_time",
            "end_time",
            "text",
        ],
    ):
        source = hit["_source"]

        episode_filename_prefix = source["episode_filename_prefix"]
        show_filename_prefix = source["show_filename_prefix"]
        start_time = float(source["start_time"])
        end_time = float(source["end_time"])
        text = source["text"]

        clip_id = make_clip_id(
            episode_filename_prefix=episode_filename_prefix,
            show_filename_prefix=show_filename_prefix,
            start_time=start_time,
            end_time=end_time,
        )

        chunks.append(
            ChunkDoc(
                clip_id=clip_id,
                episode_filename_prefix=episode_filename_prefix,
                show_filename_prefix=show_filename_prefix,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

        if limit is not None and len(chunks) >= limit:
            break

    return chunks


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def load_embedding_model(
    model_name: str,
    device: str = "cpu",
) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def embed_texts_with_model(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(np.asarray(embeddings))


def retrieve_top_k_with_loaded_model(
    query: str,
    model: SentenceTransformer,
    chunk_docs: list[ChunkDoc],
    chunk_embeddings: np.ndarray,
    top_k: int,
) -> list[RetrievedResult]:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    query_embedding = normalize_embeddings(np.asarray(query_embedding))[0]

    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]

    results: list[RetrievedResult] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunk_docs[idx]
        results.append(
            RetrievedResult(
                clip_id=chunk.clip_id,
                score=float(scores[idx]),
                rank=rank,
                text=chunk.text,
                episode_filename_prefix=chunk.episode_filename_prefix,
                show_filename_prefix=chunk.show_filename_prefix,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
        )

    return results


JUDGE_SYSTEM_PROMPT = """
You are a strict information retrieval relevance judge.

You will receive:
1. a search query
2. one retrieved podcast transcript chunk

Score relevance using:
0 = not relevant
1 = somewhat relevant
2 = highly relevant

Return ONLY valid JSON in this exact format:
{
  "relevance": 0,
  "reason": "short explanation"
}

Judge based on actual semantic relevance, not only keyword overlap.
Be strict.
""".strip()


def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract JSON from judge output:\n{text}")

    return json.loads(match.group(0))


def clamp_relevance(score: int) -> int:
    return max(0, min(2, score))


def judge_chunk_with_ollama_once(
    query: str,
    chunk_text: str,
    judge_model: str = "gpt-oss:120b-cloud",
    temperature: float = 0.1,
) -> tuple[int, str]:
    user_prompt = f"""Query:
{query}

Retrieved transcript chunk:
{chunk_text}

Return JSON only.
"""

    response = chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )

    content = response["message"]["content"]
    parsed = extract_json_from_text(content)

    relevance = int(parsed["relevance"])
    reason = str(parsed.get("reason", "")).strip()
    relevance = clamp_relevance(relevance)

    return relevance, reason


def judge_chunk_with_ollama(
    query: str,
    chunk_text: str,
    judge_model: str = "gpt-oss:120b-cloud",
    num_judgments: int = 3,
    temperature: float = 0.1,
) -> tuple[int, str, list[int], list[str], float]:
    scores: list[int] = []
    reasons: list[str] = []

    for _ in range(num_judgments):
        relevance, reason = judge_chunk_with_ollama_once(
            query=query,
            chunk_text=chunk_text,
            judge_model=judge_model,
            temperature=temperature,
        )
        scores.append(relevance)
        reasons.append(reason)

    average_score = sum(scores) / len(scores)
    final_score = int(round(average_score))
    final_score = clamp_relevance(final_score)

    combined_reason = " | ".join(
        f"judge_{i + 1}: {reason}" for i, reason in enumerate(reasons)
    )

    return final_score, combined_reason, scores, reasons, average_score


def judge_retrieved_results(
    query: str,
    model_name: str,
    retrieved_results: list[RetrievedResult],
    judge_model: str = "gpt-oss:120b-cloud",
    num_judgments: int = 3,
    temperature: float = 0.1,
) -> list[JudgedResult]:
    judged: list[JudgedResult] = []

    for result in retrieved_results:
        final_relevance, combined_reason, scores, reasons, average_score = (
            judge_chunk_with_ollama(
                query=query,
                chunk_text=result.text,
                judge_model=judge_model,
                num_judgments=num_judgments,
                temperature=temperature,
            )
        )

        judged.append(
            JudgedResult(
                query=query,
                model_name=model_name,
                clip_id=result.clip_id,
                rank=result.rank,
                retrieval_score=result.score,
                relevance=final_relevance,
                judge_reason=combined_reason,
                text=result.text,
                episode_filename_prefix=result.episode_filename_prefix,
                show_filename_prefix=result.show_filename_prefix,
                start_time=result.start_time,
                end_time=result.end_time,
                judge_scores=scores,
                judge_reasons=reasons,
                judge_average_score=average_score,
            )
        )

    return judged