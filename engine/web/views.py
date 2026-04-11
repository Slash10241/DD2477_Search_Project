from time import perf_counter
from enum import Enum
from typing import Callable

from django.shortcuts import render
from django.http import HttpRequest

from .services.hybrid_search import hybrid_search
from .services.lexical_search import lexical_search
from .services.postprocessing import rerank_and_summarize
from .services.metadata_lookup import enrich_results_with_metadata
from .services.vector_search import vector_search

class SearchMode(str, Enum):
    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"

DEFAULT_SEARCH_MODE = SearchMode.LEXICAL
VALID_SEARCH_MODES = frozenset(mode.value for mode in SearchMode)
SEARCH_EXECUTORS: dict[SearchMode, Callable[[str], list]] = {
    SearchMode.LEXICAL: lexical_search,
    SearchMode.VECTOR: lambda query: vector_search(query, num_candidates=100),
    SearchMode.HYBRID: lambda query: hybrid_search(query, num_candidates=100),
}

def parse_search_mode(raw_mode: str | None) -> SearchMode:
    normalized_mode = (raw_mode or DEFAULT_SEARCH_MODE.value).strip().lower()
    if normalized_mode in VALID_SEARCH_MODES:
        return SearchMode(normalized_mode)
    return DEFAULT_SEARCH_MODE

def index(request: HttpRequest):
    return render(
        request,
        "index.html",
        {
            "mode": DEFAULT_SEARCH_MODE.value,
            "summary_text": "",
            "retrieval_duration_ms": None,
        },
    )

def search(request: HttpRequest):
    q = (request.GET.get("q") or "").strip()
    mode: SearchMode = parse_search_mode(request.GET.get("mode"))

    results = []
    summary_text = ""
    retrieval_duration_ms: float | None = None
    error_message = ""
    if q:
        started_at = perf_counter()
        try:
            results = SEARCH_EXECUTORS[mode](q)

            results = enrich_results_with_metadata(results)
            postprocess_output = rerank_and_summarize(results)
            results = postprocess_output["results"]
            summary_text = postprocess_output["summary"]
        except Exception as exc:
            error_message = str(exc)
        finally:
            retrieval_duration_ms = (perf_counter() - started_at) * 1000.0

    template_name = (
        "partials/results.html"
        if getattr(request, "htmx", False)
        else "index.html"
    )
    return render(
        request,
        template_name,
        {
            "q": q,
            "mode": mode.value,
            "results": results,
            "summary_text": summary_text,
            "retrieval_duration_ms": retrieval_duration_ms,
            "error_message": error_message,
        },
    )
