from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("search", views.search, name="search"),
    path("llm/highlight", views.highlight_results, name="highlight_results"),
    path("llm/summarize", views.summarize_results, name="summarize_results"),
]