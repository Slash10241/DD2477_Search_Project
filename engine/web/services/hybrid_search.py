from collections import defaultdict
from elasticsearch.dsl import Search

from .elastic_utils import SearchResult, search, source
from .lexical_search import lexical_query_builder
from .vector_search import vector_query_builder

USE_BUILTIN_RRF = False


def hybrid_query_builder(
	s: Search,
	q: str,
	num_candidates: int,
	k: int,
	max_results: int,
	rrf_rank_constant: int = 60,
	rrf_window_size: int = 100,
) -> Search:
	s = lexical_query_builder(s, q)
	s = vector_query_builder(s, q, k, num_candidates, max_results)
	return s.rank(
		rrf={
			"rank_constant": rrf_rank_constant,
			"rank_window_size": rrf_window_size,
		}
	)


def _doc_key(hit) -> tuple:
    return (
        hit.episode_filename_prefix,
        hit.show_filename_prefix,
        hit.start_time,
        hit.end_time,
    )


def _rrf_score(rank: int, rank_constant: int) -> float:
    return 1.0 / (rank_constant + rank)


def _hit_to_result(hit, score: float) -> SearchResult:
    return {
        "score": float(score),
        "source": {
            "text": hit.text,
            "episode_filename_prefix": hit.episode_filename_prefix,
            "show_filename_prefix": hit.show_filename_prefix,
            "start_time": hit.start_time,
            "end_time": hit.end_time,
        },
    }


def hybrid_search(
    query_text: str,
    num_candidates: int,
    k: int = 20,
    max_results: int = 20,
    rrf_rank_constant: int = 60,
    rrf_window_size: int = 100,
) -> list[SearchResult]:
    query = query_text.strip()
    if not query or k <= 0 or max_results <= 0:
        return []
    
    if USE_BUILTIN_RRF:
        return [
            _hit_to_result(hit, hit.meta.score)
            for hit in (
                hybrid_query_builder(
                    search(),
                    query,
                    num_candidates,
                    k,
                    max_results,
                    rrf_rank_constant,
                    rrf_window_size,
                )
                .source(source())[:max_results]
            ).execute()
        ]
    
    # Run lexical search
    lexical_search_query = (
        lexical_query_builder(search(), query)
        .source(source())[:rrf_window_size]
    )
    lexical_hits = lexical_search_query.execute()

    # Run vector search
    vector_search_query = (
        vector_query_builder(search(), query, k, num_candidates, max_results)
        .source(source())[:rrf_window_size]
    )
    vector_hits = vector_search_query.execute()

    # RRF fusion
    scores = defaultdict(float)
    hits_by_key = {}

    for rank, hit in enumerate(lexical_hits, start=1):
        doc_key = _doc_key(hit)
        scores[doc_key] += _rrf_score(rank, rrf_rank_constant)
        hits_by_key[doc_key] = hit

    for rank, hit in enumerate(vector_hits, start=1):
        doc_key = _doc_key(hit)
        scores[doc_key] += _rrf_score(rank, rrf_rank_constant)
        if doc_key not in hits_by_key:
            hits_by_key[doc_key] = hit

    # Sort fused results
    top_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )[:max_results]

    return [
        _hit_to_result(hits_by_key[key], scores[key])
        for key in top_keys
    ]
