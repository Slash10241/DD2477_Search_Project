from elasticsearch.dsl import Search

from .elastic_utils import SearchResult, search, source
from .lexical_search import lexical_query_builder
from .vector_search import vector_query_builder

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

def hybrid_search(
	query_text: str,
	num_candidates: int,
	k: int = 20,
	max_results: int = 20,
	rrf_rank_constant: int = 60,
	rrf_window_size: int = 100,
) -> list[SearchResult]:
	q = query_text.strip()
	if not q or k <= 0 or max_results <= 0:
		return []

	s = (
		hybrid_query_builder(
			search(),
			q,
			num_candidates,
			k,
			max_results,
			rrf_rank_constant,
			rrf_window_size,
		)
		.source(source())[:max_results]
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
