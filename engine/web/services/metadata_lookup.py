import csv
from dotenv import load_dotenv
import os
from pathlib import Path
from threading import Lock
from typing import TypedDict
from .elastic_utils import SearchResultWithOptionalMetadata, SearchResultWithOptionalMetadata

load_dotenv()
class ShowMetadata(TypedDict):
    show_name: str
    episodes: dict[str, str]

_METADATA_LOCK = Lock()
_METADATA_CACHE: dict[str, ShowMetadata] | None = None

def _metadata_tsv_path() -> Path | None:
    configured_path = os.environ.get("METADATA_TSV_PATH")
    if configured_path:
        return Path(configured_path)

def _load_metadata_map() -> dict[str, ShowMetadata]:
    tsv_path = _metadata_tsv_path()
    if not tsv_path or not tsv_path.exists():
        return {}

    metadata_map: dict[str, ShowMetadata] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            show_prefix = (row.get("show_filename_prefix") or "").strip()
            episode_prefix = (row.get("episode_filename_prefix") or "").strip()
            if not show_prefix or not episode_prefix:
                continue

            show_entry = metadata_map.setdefault(
                show_prefix,
                {
                    "show_name": (row.get("show_name") or "").strip(),
                    "episodes": {},
                },
            )

            # Keep the first non-empty show name in case the metadata contains mixed rows.
            if not show_entry["show_name"]:
                show_entry["show_name"] = (row.get("show_name") or "").strip()

            show_entry["episodes"][episode_prefix] = (row.get("episode_name") or "").strip()

    return metadata_map

def _get_metadata_map() -> dict[str, ShowMetadata]:
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        with _METADATA_LOCK:
            if _METADATA_CACHE is None:
                _METADATA_CACHE = _load_metadata_map()
    return _METADATA_CACHE

def warm_metadata_cache() -> int:
    return len(_get_metadata_map())

def enrich_results_with_metadata(results: list[SearchResultWithOptionalMetadata]) -> list[SearchResultWithOptionalMetadata]:
    if not results:
        return results

    metadata_map = _get_metadata_map()
    if not metadata_map:
        return results
    
    def get_result_with_metadata(res: SearchResultWithOptionalMetadata) -> SearchResultWithOptionalMetadata:
        show_prefix = str(res["source"]["show_filename_prefix"]).strip()
        episode_prefix = str(res["source"]["episode_filename_prefix"]).strip()
        if not show_prefix or not episode_prefix:
            return res

        show_metadata = metadata_map.get(show_prefix)
        if not show_metadata:
            return res
        episode_name = show_metadata["episodes"].get(episode_prefix, "")

        if show_metadata["show_name"]:
            enriched_res: SearchResultWithOptionalMetadata = {
                "score": res["score"],
                "source": res["source"],
                "show_name": show_metadata["show_name"]
            }
            if episode_name:
                enriched_res["episode_name"] = episode_name
            
            return enriched_res
        
        return res
    
    return list(map(get_result_with_metadata, results,))
