from threading import Lock

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
VECTOR_DIM = 768

_model_lock = Lock()
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _model_lock:
        if _embedding_model is not None:
            return _embedding_model

        _embedding_model = SentenceTransformer(MODEL_NAME)
        return _embedding_model


def preload_embedding_model() -> bool:
    try:
        get_embedding_model()
        return True
    except Exception:
        return False


def get_query_embedding_vector(query_text: str) -> list[float]:
    q = query_text.strip()
    if not q:
        return [0.0] * VECTOR_DIM

    vector = get_embedding_model().encode(q)
    return vector.tolist()
