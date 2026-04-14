from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_DIM = 768


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def get_query_embedding_vector(query_text: str) -> list[float]:
    q = query_text.strip()
    if not q:
        return [0.0] * VECTOR_DIM

    vector = get_embedding_model().encode(q)
    return vector.tolist()
