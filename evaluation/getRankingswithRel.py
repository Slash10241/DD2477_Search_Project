import argparse
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

from search.hybrid_search import hybrid_search
from search.vector_search import vector_search
from search.lexical_search import lexical_search
from search.metadata_lookup import enrich_results_with_metadata, warm_metadata_cache

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure it is set in your .env file.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')

MAX_RESULTS = 20
NUM_CANDIDATES = 200
K = 20

# Structured as: rel_dict[query][show][episode][start_time] = rating
rel_dict = {}

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

# ── Gemini Integration ───────────────────────────────────────────────────────

def get_relevance_rating(query: str, content: str) -> str:
    prompt = f"""
    Analyze the relevance of the following podcast transcript snippet to the user query.
    Rate it using ONLY the integer. Give output just a single integer and nothing else:
    0 if query not mentioned
    1 if only mentioned but not discussed
    2 if discussed but not extensively
    3 if discussed extensively

    Query: {query}
    Transcript: {content}
    Rating:"""
    
    MAX_RETRIES = 100000
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            rating = response.text.strip()[0] 
            if rating in "0123":
                return rating
            continue
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ! Rate limit or error encountered. Retrying in 30s... ({e})")
                time.sleep(30)
                continue
            else:
                return "Error"

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_result(rank: int, result, relevance_score: str) -> str:
    show_name    = result.get("show_name", result["source"]["show_filename_prefix"])
    episode_name = result.get("episode_name", result["source"]["episode_filename_prefix"])
    text         = result["source"]["text"]
    start        = format_timestamp(result["source"]["start_time"])
    end          = format_timestamp(result["source"]["end_time"])

    lines = [
        f"[{rank}]",
        f"Relevance: {relevance_score}/3",
        f"Podcast  : {show_name}",
        f"Episode  : {episode_name}",
        f"Content  : {text}",
        f"Time     : {start} – {end}",
        "",
    ]
    return "\n".join(lines)

def run_search(query_text: str, mode: str) -> list:
    if mode == "hybrid":
        return hybrid_search(query_text=query_text, num_candidates=NUM_CANDIDATES, k=K, max_results=MAX_RESULTS)
    elif mode == "vector":
        return vector_search(query_text=query_text, num_candidates=NUM_CANDIDATES, k=K, max_results=MAX_RESULTS)
    elif mode == "lexical":
        return lexical_search(query_text=query_text, max_results=MAX_RESULTS)
    return []

# ── Main Logic ────────────────────────────────────────────────────────────────

def run_all_queries(mode: str) -> None:
    os.makedirs("query_outputs", exist_ok=True)
    output_file = f"query_outputs/query_results_{mode}.txt"

    print(f"--- Starting Search Mode: {mode} ---")
    warm_metadata_cache()

    with open(output_file, "w", encoding="utf-8") as fh:
        for query_text in QUERIES:
            print(f"Querying: {query_text!r}")
            raw_results = run_search(query_text, mode)
            enriched = enrich_results_with_metadata(raw_results)
            num_found = len(enriched)

            fh.write(f"QUERY: {query_text}\nRESULTS FOUND: {num_found}\n" + "-"*40 + "\n")

            for rank, result in enumerate(enriched, start=1):
                # ── Dictionary Lookup Logic ──
                show = result.get("show_name", result["source"]["show_filename_prefix"])
                ep = result.get("episode_name", result["source"]["episode_filename_prefix"])
                t_start = result["source"]["start_time"]
                
                # Navigate nested dict safely
                cached_val = rel_dict.get(query_text, {}).get(show, {}).get(ep, {}).get(t_start)

                if cached_val is not None:
                    relevance = cached_val
                    print(f"  [{rank}/{num_found}] Using cached rating: {relevance}")
                else:
                    content_snippet = result["source"]["text"]
                    relevance = get_relevance_rating(query_text, content_snippet)
                    
                    # Store in dict for future search modes/loops
                    rel_dict.setdefault(query_text, {}).setdefault(show, {}).setdefault(ep, {})[t_start] = relevance
                    print(f"  [{rank}/{num_found}] Gemini rated: {relevance}")

                fh.write(format_result(rank, result, relevance))

            fh.write("\n" + "="*60 + "\n\n")

    print(f"Results saved to '{output_file}'.\n")

if __name__ == "__main__":
    # Corrected the loop to iterate through each mode
    for mode in SEARCH_MODES:
        run_all_queries(mode=mode)