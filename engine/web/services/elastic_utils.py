from typing import Any, TypedDict
from elasticsearch import Elasticsearch
from elasticsearch.dsl import Search

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"
ES_CLIENT = Elasticsearch([ES_HOST])

def search():
    return Search(using=ES_CLIENT, index=INDEX_NAME)

class ResultSource(TypedDict):
    text: str
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float

def source() -> list[str | Any]:
    return list(ResultSource.__required_keys__)
