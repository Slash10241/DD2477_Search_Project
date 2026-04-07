from elasticsearch.dsl import Search
from .elastic_utils import SearchResult, search, source

VECTOR_FIELD = "text_embedding"
VECTOR_DIM = 384


def build_dummy_query_vector(query_text: str, dim: int = VECTOR_DIM) -> list[float]:
	"""Build a deterministic dummy vector from query text for development/testing."""
	q = query_text.strip()
	if not q:
		return [0.0] * dim

	seed = sum(ord(ch) for ch in q)
	return [((seed + i) % 97) / 97.0 for i in range(dim)]

def vector_query_builder(
	s: Search,
	q: str,
	k: int,
	num_candidates: int,
	max_results: int,
):
	effective_k = max(k, max_results)
	return s.knn(
		field=VECTOR_FIELD,
		query_vector=build_dummy_query_vector(q),
		k=effective_k,
		num_candidates=max(num_candidates, effective_k),
	)

def vector_search(
	query_text: str,
	num_candidates: int,
	k: int = 20,
	max_results: int = 20,
) -> list[SearchResult]:
	q = query_text.strip()
	if not q or k <= 0 or max_results <= 0:
		return []

	s = (
		vector_query_builder(search(), q, k, num_candidates, max_results)
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
