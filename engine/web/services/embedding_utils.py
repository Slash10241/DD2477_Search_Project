VECTOR_DIM = 768

def build_dummy_query_vector(query_text: str, dim: int = VECTOR_DIM) -> list[float]:
	"""Build a deterministic dummy vector from query text for development/testing."""
	q = query_text.strip()
	if not q:
		return [0.0] * dim

	seed = sum(ord(ch) for ch in q)
	return [((seed + i) % 97) / 97.0 for i in range(dim)]

def get_query_embedding_vector(query_text: str) -> list[float]:
	return build_dummy_query_vector(query_text, VECTOR_DIM)
