from typing import TypedDict

from elasticsearch.dsl import Search, query
from .elastic_utils import search, SearchResult, source


def lexical_query_builder(s: Search, query_text: str):
    return s.query(query.Match(text=query_text))

def lexical_search(query_text: str, max_results: int = 20) -> list[SearchResult]:
    q = query_text.strip()
    if not q:
        return []
    
    s = (
        lexical_query_builder(search(), q)
        .source(source())
        [:max_results]
    )
    
    response = s.execute()

    return [
        {
            "score": float(hit.meta.score),
            "source": {
                "text": hit.text,
                "episode_filename_prefix": hit.episode_filename_prefix,
                "show_filename_prefix": hit.show_filename_prefix,
                "start_time": hit.start_time,
                "end_time": hit.end_time,
            },
        }
        for hit in response
    ]
