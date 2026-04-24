from django.conf import settings

from .llm_utils import generate_content
from .elastic_utils import SearchResultWithOptionalMetadata

def generate_rag_answer(user_question: str, query: str, results: list[SearchResultWithOptionalMetadata]) -> str:
    model_name = settings.GEMINI_MODEL

    context_blocks = []
    for idx, item in enumerate(results, start=1):
        show_name = item.get("show_name") or item["source"].get("show_filename_prefix", "")
        episode_name = item.get("episode_name") or item["source"].get("episode_filename_prefix", "")
        text = item["source"].get("text", "")

        context_blocks.append(
            f"Result {idx}\n"
            f"Show: {show_name}\n"
            f"Episode: {episode_name}\n"
            f"Text:\n{text}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are answering a user question using retrieved podcast transcript chunks as context.

Original retrieval query:
{query}

User question:
{user_question}

Instructions:
- Answer only using the provided context
- Do not invent information
- If the context is insufficient, say so clearly
- Combine information across chunks when useful
- Be concise but informative

Context:
{context}
"""

    response = generate_content(
        model=model_name,
        contents=prompt,
    )

    return (response.text or "").strip()
