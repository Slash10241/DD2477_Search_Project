from django.shortcuts import render
from django.http import HttpRequest

def index(request: HttpRequest):
    return render(request, "index.html")

def search(request: HttpRequest):
    q = request.GET.get("q", "").strip()

    # Temporary placeholder results; replace with Elasticsearch call
    results = []
    if q:
        results = [
            {"title": f"Result for {q} #1", "snippet": "Snippet 1"},
            {"title": f"Result for {q} #2", "snippet": "Snippet 2"},
        ]

    template_name = "partials/results.html" if getattr(request, "htmx", False) else "index.html"
    return render(request, template_name, {"q": q, "results": results})