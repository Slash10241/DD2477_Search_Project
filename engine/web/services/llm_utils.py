import os
from google import genai
from django.conf import settings

DEFAULT_MODEL = "gemini-2.5-flash-lite"


def get_api_key() -> str:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return api_key


def get_client():
    return genai.Client(api_key=get_api_key())


def get_model_name() -> str:
    return settings.GEMINI_MODEL or DEFAULT_MODEL
