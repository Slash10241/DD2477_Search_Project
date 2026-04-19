from google import genai
from django.conf import settings
import os

def _get_client():
    return genai.Client(api_key=settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY"))

def generate_summary(query: str, results: list) -> str:
    client = _get_client()

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

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
    )

    return (response.text or "").strip()