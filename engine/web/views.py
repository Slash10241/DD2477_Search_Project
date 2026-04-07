from time import perf_counter

from django.shortcuts import render
from django.http import HttpRequest

from .services.hybrid_search import hybrid_search
from .services.lexical_search import lexical_search
from .services.postprocessing import rerank_and_summarize
from .services.metadata_lookup import enrich_results_with_metadata
from .services.vector_search import vector_search

VALID_SEARCH_MODES = {"lexical", "vector", "hybrid"}
DEFAULT_SEARCH_MODE = "lexical"

def index(request: HttpRequest):
    return render(
        request,
        "index.html",
        {
            "mode": DEFAULT_SEARCH_MODE,
            "summary_text": "",
            "retrieval_duration_ms": None,
        },
    )

def search(request: HttpRequest):
    q = (request.GET.get("q") or "").strip()
    requested_mode = (request.GET.get("mode") or DEFAULT_SEARCH_MODE).strip().lower()
    mode = requested_mode if requested_mode in VALID_SEARCH_MODES else DEFAULT_SEARCH_MODE

    results = []
    summary_text = ""
    retrieval_duration_ms: float | None = None
    error_message = ""
    if q:
        started_at = perf_counter()
        try:
            if mode == "vector":
                results = vector_search(q, num_candidates=100)
            elif mode == "hybrid":
                results = hybrid_search(q, num_candidates=100)
            else:
                results = lexical_search(q)

            results = enrich_results_with_metadata(results)
            postprocess_output = rerank_and_summarize(results)
            results = postprocess_output["results"]
            summary_text = postprocess_output["summary"]
        except Exception as exc:
            error_message = str(exc)
        finally:
            retrieval_duration_ms = (perf_counter() - started_at) * 1000.0

    template_name = "partials/results.html" if getattr(request, "htmx", False) else "index.html"
    return render(
        request,
        template_name,
        {
            "q": q,
            "mode": mode,
            "results": results,
            "summary_text": summary_text,
            "retrieval_duration_ms": retrieval_duration_ms,
            "error_message": error_message,
        },
    )
