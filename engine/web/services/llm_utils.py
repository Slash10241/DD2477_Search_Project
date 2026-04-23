import os
import logging
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfigOrDict
from django.conf import settings

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_31_FALLBACK_MODEL = "gemini-2.5-flash-lite"

logger = logging.getLogger(__name__)


def get_api_key() -> str:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return api_key


def get_backup_api_key() -> str:
    return getattr(settings, "GEMINI_BACKUP_API_KEY", "") or os.environ.get(
        "GEMINI_BACKUP_API_KEY", ""
    )


def get_client():
    return genai.Client(api_key=get_api_key())


def get_model_name() -> str:
    return settings.GEMINI_MODEL or DEFAULT_MODEL


def _is_gemini_31_model(model_name: str) -> bool:
    return model_name.startswith("gemini-3.1")


def _get_31_fallback_model() -> str:
    return (
        getattr(settings, "GEMINI_31_FALLBACK_MODEL", "")
        or os.environ.get("GEMINI_31_FALLBACK_MODEL", "")
        or DEFAULT_31_FALLBACK_MODEL
    )


def _extract_http_status(e: Exception) -> int | None:
    # SDK errors expose status under different attribute names depending on version.
    for attr in ("status_code", "code", "status"):
        value = getattr(e, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(e, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code

    return None


def generate_content(
    model: str,
    contents: Any,
    config: GenerateContentConfigOrDict | None = None,
):
    primary_key = get_api_key()
    backup_key = get_backup_api_key()

    current_key = primary_key
    current_model = model

    did_model_fallback = False
    # False if we have a backup key, True otherwise
    using_backup_key = not bool(backup_key)

    while True:
        client = genai.Client(api_key=current_key)

        try:
            return client.models.generate_content(
                model=current_model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            status_code = _extract_http_status(e)

            if (
                status_code == 503
                and _is_gemini_31_model(current_model)
                and not did_model_fallback
            ):
                fallback_model = _get_31_fallback_model()
                logger.warning(
                    "Gemini request got 503; retrying with fallback model '%s' instead of '%s'.",
                    fallback_model,
                    current_model,
                )
                current_model = fallback_model
                did_model_fallback = True
                continue

            if status_code == 429 and not using_backup_key:
                logger.warning(
                    "Gemini request got 429; retrying with backup API key using model '%s'.",
                    current_model,
                )
                current_key = backup_key
                using_backup_key = True
                continue

            raise
