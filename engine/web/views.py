from time import perf_counter
from enum import Enum
from typing import Callable
import json

from django.shortcuts import render
from django.http import HttpRequest, JsonResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from .services.hybrid_search import hybrid_search
from .services.lexical_search import lexical_search
from .services.postprocessing import rerank_and_summarize
from .services.metadata_lookup import enrich_results_with_metadata
from .services.vector_search import vector_search
from .services.llm_highlight import highlight_results_in_batches


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


def _run_search_pipeline(q: str, mode: SearchMode) -> tuple[list, str]:
    results = SEARCH_EXECUTORS[mode](q)
    results = enrich_results_with_metadata(results)
    postprocess_output = rerank_and_summarize(results)
    return list(postprocess_output["results"]), postprocess_output["summary"]


@require_GET
@ensure_csrf_cookie
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


@require_GET
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
            results, summary_text = _run_search_pipeline(q, mode)
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


@require_POST
def highlight_results(request: HttpRequest):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    q = (payload.get("q") or "").strip()
    mode = parse_search_mode(payload.get("mode"))
    top_k_raw = payload.get("top_k", 5)

    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid top_k value."}, status=400)

    if not q:
        return JsonResponse({"error": "Query is required."}, status=400)

    try:
        results, _summary_text = _run_search_pipeline(q, mode)
        selected_results = list(results[:top_k])

        highlighted_results = highlight_results_in_batches(
            query_text=q,
            results=selected_results,
            batch_size=5,
        )

        return JsonResponse(
            {
                "results": highlighted_results,
                "top_k": top_k,
                "mode": mode.value,
            }
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    
@require_POST
def summarize_results(request: HttpRequest):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    q = (payload.get("q") or "").strip()
    mode = parse_search_mode(payload.get("mode"))
    top_k = min(int(payload.get("top_k", 10)), 20)

    if not q:
        return JsonResponse({"error": "Query is required."}, status=400)

    try:
        results, _ = _run_search_pipeline(q, mode)
        selected_results = list(results[:top_k])

        from .services.llm_summary import generate_summary

        summary = generate_summary(q, selected_results)

        return JsonResponse({"summary": summary})
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)