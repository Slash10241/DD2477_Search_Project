from typing import Any, TypedDict, NotRequired, Required
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

class SearchResultWithOptionalMetadata(TypedDict, total=False):
    score: Required[float]
    source: Required[ResultSource]
    show_name: NotRequired[str]
    episode_name: NotRequired[str]

SearchResult = SearchResultWithOptionalMetadata

class LLMEnrichedSearchResult(SearchResultWithOptionalMetadata, total=False):
    highlighted_text: Required[str]


def source() -> list[str | Any]:
    return list(ResultSource.__required_keys__)
