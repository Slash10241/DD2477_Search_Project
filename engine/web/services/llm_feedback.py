from __future__ import annotations

import os
from copy import deepcopy
from math import log2
from typing import Sequence

from django.conf import settings
from google import genai
from pydantic import BaseModel, Field, ValidationError

from .elastic_utils import SearchResult

DEFAULT_FEEDBACK_MODEL = "gemini-3.1-flash-lite-preview"


class FeedbackItem(BaseModel):
    result_index: int
    relevance: int = Field(ge=0, le=3)


class FeedbackResponse(BaseModel):
    feedback: list[FeedbackItem] = Field(default_factory=list)


def _get_api_key() -> str:
    api_key = getattr(settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return api_key


def _get_model_name() -> str:
    return (
        getattr(settings, "GEMINI_FEEDBACK_MODEL", "")
        or os.environ.get("GEMINI_FEEDBACK_MODEL", "")
        or DEFAULT_FEEDBACK_MODEL
    )


def _get_client() -> genai.Client:
    return genai.Client(api_key=_get_api_key())


def _build_prompt(query_text: str, results: Sequence[SearchResult]) -> str:
    lines = [
        "You are given a user query and retrieved podcast transcript chunks.",
        "Assign a relevance label to each result using this exact scale:",
        "",
        "0 = not relevant",
        "1 = the query/topic is mentioned only briefly",
        "2 = there is useful detail, but the topic is not covered exhaustively",
        "3 = the topic is discussed exhaustively / strongly relevant",
        "",
        "Rules:",
        "1. Return JSON only.",
        "2. Score every result.",
        "3. Use only integers 0, 1, 2, or 3.",
        "4. Be strict and consistent.",
        "",
        f"Query: {query_text}",
        "",
        "Results:",
    ]

    for idx, item in enumerate(results):
        source = item["source"]
        text = source.get("text", "")

        lines.extend([
            f"Result {idx}:",
            "Text:",
            text,
            "",
        ])

    return "\n".join(lines)


DEFAULT_BATCH_SIZE = 10


def score_results(
    query_text: str,
    results: Sequence[SearchResult],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    if not query_text.strip() or not results:
        return {
            "results": list(results),
            "metrics": {
                "precision_at_k": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
            },
        }

    client = _get_client()
    model_name = _get_model_name()

    scored_results = [deepcopy(result) for result in results]
    labels: list[int] = []

    for i in range(0, len(scored_results), batch_size):
        batch = scored_results[i:i + batch_size]

        prompt = _build_prompt(query_text, batch)

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": FeedbackResponse.model_json_schema(),
            },
        )

        if not response.text:
            raise RuntimeError("Empty Gemini feedback response")

        try:
            parsed = FeedbackResponse.model_validate_json(response.text)
        except ValidationError as exc:
            raise RuntimeError("Invalid feedback response") from exc

        relevance_map = {
            item.result_index: item.relevance
            for item in parsed.feedback
        }

        for local_idx, result in enumerate(batch):
            relevance = relevance_map.get(local_idx, 0)
            result["feedback_relevance"] = relevance  # type: ignore
            labels.append(relevance)

    return {
        "results": scored_results,
        "metrics": _compute_metrics(labels),
    }

def _compute_metrics(labels: list[int]) -> dict[str, float]:
    if not labels:
        return {
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }

    k = len(labels)

    binary_labels = [1 if label > 0 else 0 for label in labels]
    precision_at_k = sum(binary_labels) / k

    mrr = 0.0
    for rank, rel in enumerate(binary_labels, start=1):
        if rel:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, rel in enumerate(labels, start=1):
        dcg += (2**rel - 1) / log2(rank + 1)

    ideal_labels = sorted(labels, reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal_labels, start=1):
        idcg += (2**rel - 1) / log2(rank + 1)

    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision_at_k": round(precision_at_k, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg_at_k, 4),
    }
