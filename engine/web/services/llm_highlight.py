from __future__ import annotations

import html
import logging
import os
from typing import Sequence

from django.conf import settings
from google import genai
from pydantic import BaseModel, Field, ValidationError

from .elastic_utils import SearchResultWithOptionalMetadata, LLMEnrichedSearchResult

import difflib

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_BATCH_SIZE = 5


class HighlightItem(BaseModel):
    result_index: int
    quotes: list[str] = Field(default_factory=list)

class HighlightResponse(BaseModel):
    highlights: list[HighlightItem] = Field(default_factory=list)


def _get_api_key() -> str:
    api_key = getattr(settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return api_key


def _get_model_name() -> str:
    return getattr(settings, "GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def _get_client() -> genai.Client:
    return genai.Client(api_key=_get_api_key())


def fuzzy_find(text: str, quote: str):
    text_lower = text.lower()
    quote_lower = quote.lower().strip()

    if len(quote_lower) < 4:
        return None

    best_match = None
    best_score = 0
    window_size = len(quote_lower)

    # step size → faster
    step = max(1, window_size // 4)

    for i in range(0, len(text_lower) - window_size + 1, step):
        window = text_lower[i:i + window_size]
        score = difflib.SequenceMatcher(None, window, quote_lower).ratio()

        if score > best_score:
            best_score = score
            best_match = (i, i + window_size)

    if best_score > 0.82:
        return best_match

    return None


def _apply_highlights_from_quotes(text: str, quotes: Sequence[str]) -> str:
    if not text:
        return ""

    lowered_text = text.lower()
    matches: list[tuple[int, int]] = []

    for quote in quotes:
        if not quote:
            continue

        quote_clean = quote.strip().lower()

        if len(quote_clean) < 4:
            continue

        #  exact match first
        start = lowered_text.find(quote_clean)

        if start != -1:
            end = start + len(quote_clean)
            matches.append((start, end))
            continue

        #  fuzzy fallback
        fuzzy_match = fuzzy_find(text, quote_clean)
        if fuzzy_match:
            matches.append(fuzzy_match)

    if not matches:
        return html.escape(text)

    # merge overlaps
    matches.sort()
    merged = []

    for s, e in matches:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])

    # build html
    parts = []
    cursor = 0

    for s, e in merged:
        if cursor < s:
            parts.append(html.escape(text[cursor:s]))

        parts.append(f"<mark>{html.escape(text[s:e])}</mark>")
        cursor = e

    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    return "".join(parts)


def _build_prompt(query_text: str, batch: Sequence[SearchResultWithOptionalMetadata]) -> str:
    lines = [
        "You are given a query and transcript chunks.",
        "Extract the most relevant SHORT phrases from each chunk.",
        "",
        "Rules:",
        "1. Copy phrases EXACTLY from the text.",
        "2. Do NOT paraphrase.",
        "3. Return 1-3 phrases per result.",
        "4. Each phrase should be 5–15 words.",
        "5. If not relevant, return empty list.",
        "",
        f"Query: {query_text}",
        "",
        "Results:",
    ]

    for idx, item in enumerate(batch):
        source = item["source"]
        text = source["text"]

        lines.extend([
            f"Result {idx}:",
            "Text:",
            text,
            "",
        ])

    return "\n".join(lines)


def _extract_batch_highlights(
    client: genai.Client,
    model_name: str,
    query_text: str,
    batch: Sequence[SearchResultWithOptionalMetadata],
) -> HighlightResponse:
    prompt = _build_prompt(query_text, batch)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": HighlightResponse.model_json_schema(),
        },
    )

    if not response.text:
        raise RuntimeError("Empty Gemini response")

    try:
        return HighlightResponse.model_validate_json(response.text)
    except ValidationError:
        logger.exception("Invalid Gemini response")
        raise RuntimeError("Invalid highlight response")


def _to_llm_enriched_result(result: SearchResultWithOptionalMetadata, highlighted = "") -> LLMEnrichedSearchResult:
    enriched_result: LLMEnrichedSearchResult = {
        "score": result["score"],
        "source": result["source"],
        "highlighted_text": highlighted
    }

    if "show_name" in result:
        enriched_result["show_name"] = result["show_name"]

    if "episode_name" in result:
        enriched_result["episode_name"] = result["episode_name"]

    return enriched_result


def highlight_results_in_batches(
    query_text: str,
    results: list[SearchResultWithOptionalMetadata],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[LLMEnrichedSearchResult]:
    if not query_text.strip() or not results:
        return [_to_llm_enriched_result(r) for r in results]

    client = _get_client()
    model_name = _get_model_name()
    logger.warning("Gemini highlight model in use: %s", model_name)

    res: list[LLMEnrichedSearchResult] = []
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]

        llm_output = _extract_batch_highlights(
            client,
            model_name,
            query_text,
            batch,
        )

        quote_map = {
            item.result_index: item.quotes
            for item in llm_output.highlights
        }
        res.extend(
            _to_llm_enriched_result(
                result,
                _apply_highlights_from_quotes(
                    result["source"]["text"],
                    quote_map.get(local_idx, [])
                )
            )
            for local_idx, result in enumerate(batch)
        )

    return res
