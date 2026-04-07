from typing import Any

from django import template

register = template.Library()

@register.filter(name="seconds_to_hms")
def seconds_to_hms(value: Any) -> str:
    try:
        total_seconds = int(float(value))
    except (TypeError, ValueError):
        return "00:00:00"

    if total_seconds < 0:
        total_seconds = 0

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
