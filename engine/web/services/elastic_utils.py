from typing import Any, TypedDict, NotRequired
from elasticsearch import Elasticsearch
from elasticsearch.dsl import Search
from dotenv import load_dotenv
import os

load_dotenv()

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"

API_KEY = os.environ.get("API_KEY")
ES_CLIENT = Elasticsearch(ES_HOST, api_key=API_KEY)


def search():
    return Search(using=ES_CLIENT, index=INDEX_NAME)


class ResultSource(TypedDict):
    text: str
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float
    highlight_spans: NotRequired[list[list[int]]]
    highlighted_text: NotRequired[str]


class SearchResult(TypedDict):
    score: float
    source: ResultSource
    show_name: NotRequired[str]
    episode_name: NotRequired[str]


SearchResultPossiblyWithMetadata = SearchResult


def source() -> list[str | Any]:
    return [
        "text",
        "episode_filename_prefix",
        "show_filename_prefix",
        "start_time",
        "end_time",
    ]