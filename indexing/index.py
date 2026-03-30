import argparse
from dataclasses import dataclass
import json
import os
from typing import Any, Iterator

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"

CHUNK_DURATION = 120  # 2 minutes
CHUNK_OVERLAP = 30  # overlap in seconds

# Adjust to system resources
BULK_CHUNK_SIZE = 5000
BULK_MAX_BYTES = 50 * 1024 * 1024
BULK_THREAD_COUNT = 16


@dataclass()
class Segment:
    start: float
    end: float
    text: str


@dataclass()
class Chunk:
    start_time: float
    end_time: float
    text: str


def create_index(es: Elasticsearch):
    # Create index if it does not exist
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "episode_filename_prefix": {"type": "keyword"},
                        "show_filename_prefix": {"type": "keyword"},
                        "start_time": {"type": "float"},
                        "end_time": {"type": "float"},
                        "text": {"type": "text"},
                    }
                }
            },
        )
        print(f"Created index: {INDEX_NAME}")


def parse_transcript(json_data: dict[str, Any]) -> list[Segment]:
    """
    Converts Spotify transcript JSON into a list of segments
    """
    segments: list[Segment] = []
    results = json_data.get("results", [])

    for result in results:
        for alt in result.get("alternatives", []):
            words = alt.get("words", [])
            if not words:
                continue

            # Convert word timings to seconds and collect them
            for w in words:
                # Convert "5.900s" -> float seconds
                start = float(w["startTime"].replace("s", ""))
                end = float(w["endTime"].replace("s", ""))
                text = w["word"]
                segments.append(Segment(start=start, end=end, text=text))
    return segments


def chunk_transcript(segments: list[Segment]) -> list[Chunk]:
    """
    Convert a list of segments into overlapping chunks.
    """
    chunks: list[Chunk] = []
    n = len(segments)
    start_idx = 0

    while start_idx < n:
        chunk_start_time = segments[start_idx].start
        window_end_time = chunk_start_time + CHUNK_DURATION

        # find end_idx for this chunk
        end_idx = start_idx
        while end_idx < n and segments[end_idx].start <= window_end_time:
            end_idx += 1

        if end_idx > start_idx:
            window = segments[start_idx:end_idx]
            chunks.append(
                Chunk(
                    start_time=window[0].start,
                    end_time=window[-1].end,
                    text=" ".join(seg.text for seg in window),
                )
            )

        # move start_idx forward by CHUNK_DURATION - CHUNK_OVERLAP
        next_start_time = chunk_start_time + CHUNK_DURATION - CHUNK_OVERLAP
        while start_idx < n and segments[start_idx].start < next_start_time:
            start_idx += 1

    return chunks


def iter_actions(all_files: list[str]) -> Iterator[dict[str, Any]]:
    for file_path in tqdm(all_files, desc="Indexing transcripts", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        episode_filename_prefix = os.path.splitext(os.path.basename(file_path))[0]
        show_filename_prefix = os.path.basename(os.path.dirname(file_path))

        segments = parse_transcript(data)
        chunks = chunk_transcript(segments)

        for chunk in chunks:
            yield {
                "_index": INDEX_NAME,
                "_source": {
                    "episode_filename_prefix": episode_filename_prefix,
                    "show_filename_prefix": show_filename_prefix,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "text": chunk.text,
                },
            }


def index_transcripts(es: Elasticsearch, transcripts_dir: str):
    all_files: list[str] = []
    for root, _dirs, files in os.walk(transcripts_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))

    indexed_count = 0
    for ok, _ in helpers.parallel_bulk(
        es,
        iter_actions(all_files),
        thread_count=BULK_THREAD_COUNT,
        chunk_size=BULK_CHUNK_SIZE,
        max_chunk_bytes=BULK_MAX_BYTES,
        request_timeout=120,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            indexed_count += 1

    print(f"Indexing complete. Total chunks indexed: {indexed_count}")
    print(f"Total files processed: {len(all_files)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index podcast transcripts into Elasticsearch"
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        required=True,
        help="Path to the directory containing podcast transcripts",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Elasticsearch API key",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    es = Elasticsearch(ES_HOST, api_key=args.api_key)
    create_index(es)
    index_transcripts(es, transcripts_dir=args.transcripts_dir)


if __name__ == "__main__":
    main()
