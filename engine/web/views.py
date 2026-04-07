from django.shortcuts import render
from django.http import HttpRequest

from .services.hybrid_search import hybrid_search
from .services.lexical_search import lexical_search
from .services.metadata_lookup import enrich_results_with_metadata
from .services.vector_search import vector_search

VALID_SEARCH_MODES = {"lexical", "vector", "hybrid"}
DEFAULT_SEARCH_MODE = "lexical"

def index(request: HttpRequest):
    return render(request, "index.html", {"mode": DEFAULT_SEARCH_MODE})

def search(request: HttpRequest):
    q = (request.GET.get("q") or "").strip()
    requested_mode = (request.GET.get("mode") or DEFAULT_SEARCH_MODE).strip().lower()
    mode = requested_mode if requested_mode in VALID_SEARCH_MODES else DEFAULT_SEARCH_MODE

    results = []
    error_message = ""
    if q:
        try:
            if mode == "vector":
                results = vector_search(q, num_candidates=100)
            elif mode == "hybrid":
                results = hybrid_search(q, num_candidates=100)
            else:
                results = lexical_search(q)

            results = enrich_results_with_metadata(results)
        except Exception as exc:
            error_message = str(exc)

    template_name = "partials/results.html" if getattr(request, "htmx", False) else "index.html"
    return render(
        request,
        template_name,
        {
            "q": q,
            "mode": mode,
            "results": results,
            "error_message": error_message,
        },
    )
