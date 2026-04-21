"""
Run predefined queries against the podcast search index and save results to a txt file.
Supports hybrid, vector, or lexical search. Output file is named accordingly.
"""

import argparse

from search.hybrid_search import hybrid_search
from search.vector_search import vector_search
from search.lexical_search import lexical_search
from search.metadata_lookup import enrich_results_with_metadata, warm_metadata_cache

# ── Configuration ────────────────────────────────────────────────────────────

MAX_RESULTS = 20
NUM_CANDIDATES = 200  # candidates to consider for kNN; must be >= MAX_RESULTS
K = 20                # nearest neighbours returned by kNN

SEARCH_MODES = ("hybrid", "vector", "lexical")

QUERIES = [
    "COVID-19 pandemic response and lockdown measures",
    "2020 US presidential election and voter turnout",
    "Black Lives Matter protests and racial justice",
    "Trump impeachment trial and Senate acquittal",
    "wildfires in Australia and California 2020",
    "Hong Kong protests and China national security law",
    "Brexit negotiations and the UK leaving the EU",
    "mental health during quarantine and isolation",
    "remote work and the future of the office",
    "stock market crash and economic recession 2020",
    "artificial intelligence and machine learning breakthroughs",
    "mRNA vaccines and the future of medicine",
    "climate change policy and the Paris Agreement",
    "cryptocurrency Bitcoin surge and institutional adoption",
    "privacy concerns and surveillance capitalism",
    "social media addiction and its effects on teenagers",
    "Netflix binge watching and the streaming wars",
    "Olympics postponement and athlete mental health",
    "5G technology rollout and conspiracy theories",
    "SpaceX Falcon 9 and commercial space travel",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_result(rank: int, result) -> str:
    show_name    = result.get("show_name", result["source"]["show_filename_prefix"])
    episode_name = result.get("episode_name", result["source"]["episode_filename_prefix"])
    text         = result["source"]["text"]
    start        = format_timestamp(result["source"]["start_time"])
    end          = format_timestamp(result["source"]["end_time"])

    lines = [
        f"[{rank}]",
        f"Podcast : {show_name}",
        f"Episode : {episode_name}",
        f"Content : {text}",
        f"Time    : {start} – {end}",
        "",
    ]
    return "\n".join(lines)


def run_search(query_text: str, mode: str) -> list:
    """Dispatch to the appropriate search function based on mode."""
    if mode == "hybrid":
        return hybrid_search(
            query_text=query_text,
            num_candidates=NUM_CANDIDATES,
            k=K,
            max_results=MAX_RESULTS,
        )
    elif mode == "vector":
        return vector_search(
            query_text=query_text,
            num_candidates=NUM_CANDIDATES,
            k=K,
            max_results=MAX_RESULTS,
        )
    elif mode == "lexical":
        return lexical_search(
            query_text=query_text,
            max_results=MAX_RESULTS,
        )
    else:
        raise ValueError(f"Unknown search mode: {mode!r}. Choose from {SEARCH_MODES}.")


def run_all_queries(mode: str) -> None:
    output_file = f"query_outputs/query_results_{mode}.txt"

    print(f"Search mode : {mode}")
    print(f"Output file : {output_file}")
    print("Warming metadata cache …")
    cached = warm_metadata_cache()
    print(f"  {cached} shows loaded.\n")

    with open(output_file, "w", encoding="utf-8") as fh:
        for query_text in QUERIES:
            print(f"Querying: {query_text!r}")

            raw_results = run_search(query_text, mode)
            enriched = enrich_results_with_metadata(raw_results)
            num_found = len(enriched)

            # ── Write query header ──────────────────────────────────────────
            fh.write(f"{query_text}\n")
            fh.write(f"{num_found}\n")

            for rank, result in enumerate(enriched, start=1):
                fh.write(format_result(rank, result))

            fh.write("\n")  # blank line between queries
            print(f"  → {num_found} results written.\n")

    print(f"Done. Results saved to '{output_file}'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predefined queries against the podcast search index."
    )
    parser.add_argument(
        "--mode",
        choices=SEARCH_MODES,
        default="hybrid",
        help="Search mode to use (default: hybrid).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_queries(mode=args.mode)