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
from .services.metadata_lookup import enrich_results_with_metadata
from .services.vector_search import vector_search
from .services.llm_highlight import highlight_results_in_batches
from .services.llm_feedback import score_results
from .services.llm_summary import _get_client
from django.conf import settings

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



def _generate_rag_answer(user_question: str, query: str, results: list) -> str:
    client = _get_client()
    model_name = settings.GEMINI_MODEL

    context_blocks = []
    for idx, item in enumerate(results, start=1):
        show_name = item.get("show_name") or item["source"].get("show_filename_prefix", "")
        episode_name = item.get("episode_name") or item["source"].get("episode_filename_prefix", "")
        text = item["source"].get("text", "")

        context_blocks.append(
            f"Result {idx}\n"
            f"Show: {show_name}\n"
            f"Episode: {episode_name}\n"
            f"Text:\n{text}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are answering a user question using retrieved podcast transcript chunks as context.

Original retrieval query:
{query}

User question:
{user_question}

Instructions:
- Answer only using the provided context
- Do not invent information
- If the context is insufficient, say so clearly
- Combine information across chunks when useful
- Be concise but informative

Context:
{context}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    return (response.text or "").strip()


def parse_search_mode(raw_mode: str | None) -> SearchMode:
    normalized_mode = (raw_mode or DEFAULT_SEARCH_MODE.value).strip().lower()
    if normalized_mode in VALID_SEARCH_MODES:
        return SearchMode(normalized_mode)
    return DEFAULT_SEARCH_MODE


def _run_search_pipeline(q: str, mode: SearchMode) -> tuple[list, str]:
    results = SEARCH_EXECUTORS[mode](q)
    results = enrich_results_with_metadata(results)
    return list(results), ""

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


@require_POST
def feedback_results(request: HttpRequest):
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

        scored_output = score_results(
            query_text=q,
            results=selected_results,
        )

        return JsonResponse(
            {
                "results": scored_output["results"],
                "metrics": scored_output["metrics"],
                "top_k": top_k,
                "mode": mode.value,
            }
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    


@require_POST
def ask_results(request: HttpRequest):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    q = (payload.get("q") or "").strip()
    ask_query = (payload.get("ask_query") or "").strip()
    mode = parse_search_mode(payload.get("mode"))
    top_k_raw = payload.get("top_k", 5)

    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid top_k value."}, status=400)

    if not q:
        return JsonResponse({"error": "Search query is required."}, status=400)

    if not ask_query:
        return JsonResponse({"error": "Ask query is required."}, status=400)

    try:
        results, _summary_text = _run_search_pipeline(q, mode)
        selected_results = list(results[:top_k])

        answer = _generate_rag_answer(
            user_question=ask_query,
            query=q,
            results=selected_results,
        )

        return JsonResponse(
            {
                "answer": answer,
                "top_k": top_k,
                "mode": mode.value,
            }
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)