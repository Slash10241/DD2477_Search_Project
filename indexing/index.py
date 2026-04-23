import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator
import torch
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"
CHECKPOINT_FILE = "indexed_files.txt"

CHUNK_DURATION = 120  # 2 minutes
CHUNK_OVERLAP = 30  # overlap in seconds

# Adjust to system resources
BULK_CHUNK_SIZE = 5000
BULK_MAX_BYTES = 50 * 1024 * 1024
BULK_THREAD_COUNT = 16

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH_SIZE = 64 

# For stats
HISTOGRAM_BIN_SIZE_CHARS = 500
TOP_OUTLIER_FILES = 10


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


@dataclass()
class IndexingStats:
    files_processed: int = 0
    total_chunks: int = 0

    chunk_text_len_min: int = 0
    chunk_text_len_max: int = 0
    chunk_text_len_sum: int = 0

    chunk_duration_min: float = 0.0
    chunk_duration_max: float = 0.0
    chunk_duration_sum: float = 0.0

    chunk_words_min: int = 0
    chunk_words_max: int = 0
    chunk_words_sum: int = 0

    chunk_text_len_hist_bins: dict[int, int] = field(default_factory=dict)


def _update_chunk_stats(stats: IndexingStats, chunk: Chunk):
    text_len = len(chunk.text)
    duration = max(0.0, chunk.end_time - chunk.start_time)
    word_count = len(chunk.text.split())

    if stats.total_chunks == 0:
        stats.chunk_text_len_min = text_len
        stats.chunk_text_len_max = text_len
        stats.chunk_duration_min = duration
        stats.chunk_duration_max = duration
        stats.chunk_words_min = word_count
        stats.chunk_words_max = word_count
    else:
        stats.chunk_text_len_min = min(stats.chunk_text_len_min, text_len)
        stats.chunk_text_len_max = max(stats.chunk_text_len_max, text_len)
        stats.chunk_duration_min = min(stats.chunk_duration_min, duration)
        stats.chunk_duration_max = max(stats.chunk_duration_max, duration)
        stats.chunk_words_min = min(stats.chunk_words_min, word_count)
        stats.chunk_words_max = max(stats.chunk_words_max, word_count)

    stats.total_chunks += 1
    stats.chunk_text_len_sum += text_len
    stats.chunk_duration_sum += duration
    stats.chunk_words_sum += word_count

    bin_start = (text_len // HISTOGRAM_BIN_SIZE_CHARS) * HISTOGRAM_BIN_SIZE_CHARS
    stats.chunk_text_len_hist_bins[bin_start] = (
        stats.chunk_text_len_hist_bins.get(bin_start, 0) + 1
    )


def _save_chunk_length_histogram(stats: IndexingStats, output_path: str):
    sorted_bins = sorted(stats.chunk_text_len_hist_bins.items())
    x = [bin_start for bin_start, _ in sorted_bins]
    y = [count for _, count in sorted_bins]

    plt.figure(figsize=(12, 6))
    plt.bar(x, y, width=HISTOGRAM_BIN_SIZE_CHARS * 0.9, align="edge")
    plt.title("Chunk Text Length Histogram")
    plt.xlabel("Chunk text length (characters)")
    plt.ylabel("Number of chunks")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved text length histogram: {output_path}")


def _print_indexing_stats(stats: IndexingStats):
    print("\nChunk statistics:")
    print(f"  files_processed: {stats.files_processed}")
    if stats.total_chunks == 0:
        print("  total_chunks: 0")
        return

    text_len_mean = stats.chunk_text_len_sum / stats.total_chunks
    duration_mean = stats.chunk_duration_sum / stats.total_chunks
    words_mean = stats.chunk_words_sum / stats.total_chunks

    print(f"  total_chunks: {stats.total_chunks}")
    print(
        "  chunk_text_len_chars: "
        f"min={stats.chunk_text_len_min}, "
        f"max={stats.chunk_text_len_max}, "
        f"mean={text_len_mean:.2f}"
    )
    print(
        "  chunk_duration_seconds: "
        f"min={stats.chunk_duration_min:.2f}, "
        f"max={stats.chunk_duration_max:.2f}, "
        f"mean={duration_mean:.2f}"
    )
    print(
        "  chunk_word_count: "
        f"min={stats.chunk_words_min}, "
        f"max={stats.chunk_words_max}, "
        f"mean={words_mean:.2f}"
    )


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
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine"
                        },
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
    last_start_time = -1.0

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

                # Guard against malformed replay blocks where timings restart at ~0.
                if start < last_start_time:
                    continue

                segments.append(Segment(start=start, end=end, text=text))
                last_start_time = start
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


def get_indexed_files(es: Elasticsearch) -> set[str]:
    """Query ES for all unique episode_filename_prefixes already indexed."""
    indexed = set()
    resp = es.search(
        index=INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "unique_files": {
                    "terms": {
                        "field": "episode_filename_prefix",
                        "size": 150000,
                    }
                }
            }
        }
    )
    for bucket in resp["aggregations"]["unique_files"]["buckets"]:
        indexed.add(bucket["key"])
    return indexed


def iter_actions(
    all_files: list[str], stats: IndexingStats, model, indexed_files: set[str]
) -> Iterator[dict[str, Any]]:
    for file_path in tqdm(all_files, desc="Indexing transcripts", unit="file"):
        episode_filename_prefix = os.path.splitext(os.path.basename(file_path))[0]

        if episode_filename_prefix in indexed_files:
            stats.files_processed += 1
            continue  # skip already indexed files

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = parse_transcript(data)
        stats.files_processed += 1

        chunks = chunk_transcript(segments)
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for chunk, embedding in zip(chunks, embeddings):
            _update_chunk_stats(stats, chunk)
            yield {
                "_index": INDEX_NAME,
                "_source": {
                    "episode_filename_prefix": episode_filename_prefix,
                    "show_filename_prefix": os.path.basename(os.path.dirname(file_path)),
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "text": chunk.text,
                    "embedding": embedding.tolist(),
                },
            }
         

def index_transcripts(
    es: Elasticsearch, transcripts_dir: str, max_files: int | None = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)

    all_files: list[str] = []
    for root, _dirs, files in os.walk(transcripts_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))

    total_discovered_files = len(all_files)
    if max_files is not None:
        if max_files < 1:
            raise ValueError("--max-files must be >= 1")
        all_files = all_files[:max_files]
        print(
            "File processing limit enabled: "
            f"{len(all_files)} of {total_discovered_files} files will be indexed"
        )

    indexed_files = get_indexed_files(es)
    print(f"Resuming: {len(indexed_files)} files already indexed, skipping them.")

    stats = IndexingStats()

    indexed_count = 0
    for ok, _ in helpers.parallel_bulk(
        es,
        iter_actions(all_files, stats, model, indexed_files),
        thread_count=BULK_THREAD_COUNT,
        chunk_size=BULK_CHUNK_SIZE,
        max_chunk_bytes=BULK_MAX_BYTES,
        request_timeout=120,
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            indexed_count += 1

    print(f"Indexing complete. Total new chunks indexed: {indexed_count}")
    print(f"Total files processed: {len(all_files)}")
    print(f"Total files discovered: {total_discovered_files}")
    _print_indexing_stats(stats)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    histogram_output_path = os.path.join(base_dir, "chunk_text_length_histogram.png")
    _save_chunk_length_histogram(stats, histogram_output_path)

def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_checkpoint(file_path: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(file_path + "\n")


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
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for number of files to process",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    es = Elasticsearch(ES_HOST, api_key=args.api_key)
    create_index(es)
    index_transcripts(
        es,
        transcripts_dir=args.transcripts_dir,
        max_files=args.max_files,
    )
    

def semantic_search(es, query_text, model, k=10):
    query_vec = model.encode(query_text, convert_to_numpy=True).tolist()
    resp = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vec,
            "k": k,
            "num_candidates": 100,
        },
    )
    return resp["hits"]["hits"]

if __name__ == "__main__":
    main()
