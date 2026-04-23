from django.conf import settings

from .llm_utils import generate_content

def generate_summary(query: str, results: list) -> str:
    texts = [r["source"]["text"] for r in results]

    context = "\n\n".join(texts)

    prompt = f"""
    You are given a user query and retrieved podcast transcript chunks.

    Your task is to generate a concise, well-structured summary of the information in these chunks,
    focusing on what is most relevant to the query.

    Query:
    {query}

    Instructions:
    - Focus on the parts of the text that relate to the query
    - The query may be a keyword or phrase, not a question.
    - Do not invent information
    - Avoid repetition
    - Combine information across chunks
    - Emphasize key entities, events, or concepts related to the query

    Text:
    {context}
    """

    response = generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
    )

    return (response.text or "").strip()
