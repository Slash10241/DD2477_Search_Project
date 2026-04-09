import argparse
import json
import random
import re
from dataclasses import dataclass

from elasticsearch import Elasticsearch, helpers
from ollama import chat


"""
uv run generate_eval_queries.py \
  --api-key $ES_LOCAL_API_KEY \
  --chunk-limit 3000 \
  --num-samples 30 \
  --generator-model gpt-oss:120b-cloud \
  --output-json generated_eval_queries.json
"""

ES_HOST = "http://localhost:9200"
INDEX_NAME = "podcasts"


@dataclass
class ChunkDoc:
    episode_filename_prefix: str
    show_filename_prefix: str
    start_time: float
    end_time: float
    text: str


SYSTEM_PROMPT = """
You are helping create evaluation queries for a podcast search dataset.

You will receive one podcast transcript chunk.
Your task is to generate realistic search queries for evaluating retrieval quality over a large collection of podcast chunks.

Important:
The queries must be grounded in the chunk, but they should NOT be so specific that they only match one exact snippet.
They should be broad enough that multiple chunks across the collection could plausibly be relevant.

Requirements:
- queries must be grounded in the chunk
- queries should sound like real user search queries
- queries should be moderately general, not hyper-specific
- prefer topic-level, issue-level, or discussion-level phrasing
- use neutral, professional, and presentation-appropriate wording
- avoid slang, vulgar, overly informal, or unnecessarily explicit wording
- when a chunk involves sensitive personal topics, prefer broader, higher-level, and more respectful phrasing
- prefer general themes such as relationships, communication, identity, well-being, or personal challenges over intimate or highly detailed wording when possible
- avoid phrasing that sounds sensational or unnecessarily intimate unless it is necessary for the main topic
- avoid copying long exact phrases from the chunk
- avoid exact scores, exact dates, exact quotes, or very one-off events unless central to the topic
- avoid show names, episode ids, timestamps, and speaker-specific references
- avoid strongly real-time or web-navigation style queries when possible
- avoid rare named entities unless they are central recurring concepts
- each query should plausibly retrieve more than one relevant chunk in a podcast collection

Generate exactly 6 candidate queries:
1. one short keyword-style topical query
2. one short conceptual query
3. one natural search query
4. one paraphrased topic query
5. one broader discussion-oriented query
6. one alternative wording of the main theme

Return ONLY valid JSON in this exact format:
{
  "queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4",
    "query 5",
    "query 6"
  ]
}
""".strip()


def load_chunks_from_es(
    es: Elasticsearch,
    index_name: str,
    limit: int | None = None,
) -> list[ChunkDoc]:
    chunks: list[ChunkDoc] = []

    for hit in helpers.scan(
        es,
        index=index_name,
        query={"query": {"match_all": {}}},
        _source=[
            "episode_filename_prefix",
            "show_filename_prefix",
            "start_time",
            "end_time",
            "text",
        ],
    ):
        source = hit["_source"]
        chunks.append(
            ChunkDoc(
                episode_filename_prefix=source["episode_filename_prefix"],
                show_filename_prefix=source["show_filename_prefix"],
                start_time=float(source["start_time"]),
                end_time=float(source["end_time"]),
                text=source["text"],
            )
        )

        if limit is not None and len(chunks) >= limit:
            break

    return chunks


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract JSON from model output:\n{text}")

    return json.loads(match.group(0))


def clean_query(q: str) -> str:
    q = " ".join(q.split())
    q = q.strip().strip('"').strip("'")
    q = re.sub(r"[ \t]+", " ", q)
    q = re.sub(r"[.,;:!]+$", "", q)
    return q


def deduplicate_queries(queries: list[str]) -> list[str]:
    seen = set()
    result = []

    for q in queries:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(q)

    return result


def looks_too_informal(q: str) -> bool:
    q_lower = q.lower()

    informal_terms = [
        "hookup",
        "body count",
        "hot girl",
    ]
    return any(term in q_lower for term in informal_terms)


def is_sensitive_topic(q: str) -> bool:
    q_lower = q.lower()
    sensitive_terms = [
        "sex",
        "sexual",
        "intimacy",
        "friends with benefits",
        "lgbtq",
        "transgender",
        "gay",
        "orientation",
        "gender-affirming",
    ]
    return any(term in q_lower for term in sensitive_terms)


def looks_like_live_or_web_query(q: str) -> bool:
    q_lower = q.lower()

    bad_terms = [
        "live score",
        "live scores",
        "breaking news",
        "current score",
        "watch online",
        "streaming",
        "odds",
    ]
    return any(term in q_lower for term in bad_terms)


def has_too_many_numbers(q: str) -> bool:
    return len(re.findall(r"\d+", q)) >= 3


def looks_like_exact_snippet_lookup(q: str) -> bool:
    q_lower = q.lower()

    patterns = [
        r"\bwhat happened in\b",
        r"\bwho said\b",
        r"\bexactly why\b",
        r"\b36 points\b",
        r"\btriple-double\b",
    ]

    return any(re.search(pat, q_lower) for pat in patterns)


def looks_too_entity_heavy(q: str) -> bool:
    """
    Soft heuristic:
    reject queries that look dominated by multiple uncommon named entities.
    Keep broad topical entities like Mormon, Christianity, NBA, etc.
    """
    allowed_topic_entities = {
        "mormon", "lds", "christianity", "christian", "religion",
        "lgbtq", "nba", "nbl", "podcast", "church", "faith",
        "basketball", "dating", "intimacy", "sexuality", "leadership",
        "college", "sports", "mental", "health", "identity",
        "relationships", "spiritual", "religious",
    }

    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", q)
    capitalized = [t for t in tokens if t[:1].isupper()]
    unusual_capitalized = [
        t for t in capitalized
        if t.lower() not in allowed_topic_entities
    ]

    return len(unusual_capitalized) >= 3


def normalize_question_to_search_style(q: str) -> str:
    q = clean_query(q)

    replacements = [
        (r"^how do i\s+", ""),
        (r"^how can i\s+", ""),
        (r"^what are\s+", ""),
        (r"^what is\s+", ""),
        (r"^why do\s+", ""),
        (r"^why does\s+", ""),
        (r"^how to\s+", ""),
    ]

    for pattern, repl in replacements:
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)

    q = q.strip()
    q = q[0].upper() + q[1:] if q else q
    return q


def word_count(q: str) -> int:
    return len(q.split())


def is_good_query(q: str) -> bool:
    if not q:
        return False

    wc = word_count(q)
    if wc < 2 or wc > 12:
        return False

    q_lower = q.lower()

    bad_exact = [
        "this podcast",
        "this episode",
        "the speaker says",
        "general discussion",
        "random topic",
        "something happened",
    ]
    if any(x in q_lower for x in bad_exact):
        return False

    if looks_too_informal(q):
        return False

    if looks_like_live_or_web_query(q):
        return False

    if has_too_many_numbers(q):
        return False

    if looks_like_exact_snippet_lookup(q):
        return False

    if looks_too_entity_heavy(q):
        return False

    return True


def score_query(q: str) -> int:
    """
    Prefer short-to-medium topical queries that sound like real search.
    """
    score = 0
    wc = word_count(q)

    if 2 <= wc <= 5:
        score += 3
    elif 6 <= wc <= 9:
        score += 2
    else:
        score += 1

    q_lower = q.lower()

    preferred_terms = [
        "impact", "effects", "boundaries", "communication", "leadership",
        "discipline", "faith", "religious", "dating", "intimacy",
        "basketball", "trade", "team", "performance", "mental health",
        "spiritual", "relationships", "community", "identity",
        "college", "podcast", "mission", "strategy", "culture",
    ]
    if any(term in q_lower for term in preferred_terms):
        score += 2

    weak_terms = [
        "best", "tips", "latest", "live", "highlights"
    ]
    if any(term in q_lower for term in weak_terms):
        score -= 1

    if "?" in q:
        score -= 1

    if is_sensitive_topic(q):
        score -= 1

    return score


def select_best_queries(queries: list[str], max_keep: int = 4) -> list[str]:
    queries = deduplicate_queries(queries)
    valid = [q for q in queries if is_good_query(q)]
    valid = sorted(valid, key=lambda x: (-score_query(x), word_count(x), x.lower()))

    sensitive = [q for q in valid if is_sensitive_topic(q)]
    non_sensitive = [q for q in valid if not is_sensitive_topic(q)]

    max_sensitive = max(1, int(max_keep * 0.25))
    selected = non_sensitive[: max_keep - max_sensitive] + sensitive[:max_sensitive]

    if len(selected) < max_keep:
        remaining = [q for q in valid if q not in selected]
        selected.extend(remaining[: max_keep - len(selected)])

    return selected[:max_keep]


def generate_queries_for_chunk(
    chunk: ChunkDoc,
    generator_model: str,
) -> list[str]:
    user_prompt = f"""Podcast transcript chunk:

Text:
{chunk.text}

Return JSON only.
"""

    response = chat(
        model=generator_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.35},
    )

    content = response["message"]["content"]
    parsed = extract_json(content)

    raw_queries = parsed.get("queries", [])
    if not isinstance(raw_queries, list):
        return []

    cleaned = []
    for q in raw_queries:
        if not isinstance(q, str):
            continue
        q = normalize_question_to_search_style(q)
        q = clean_query(q)
        if q:
            cleaned.append(q)

    return select_best_queries(cleaned, max_keep=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate collection-level evaluation queries from podcast chunks."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Elasticsearch API key",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="gpt-oss:120b-cloud",
        help="Ollama model used to generate candidate queries",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=3000,
        help="How many chunks to load from Elasticsearch before sampling",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="How many chunks to sample for query generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=120,
        help="Ignore chunks with too little text",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="generated_eval_queries.json",
        help="Where to save the generated queries",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    es = Elasticsearch(ES_HOST, api_key=args.api_key)

    print(f"Loading chunks from index '{INDEX_NAME}'...")
    chunks = load_chunks_from_es(es=es, index_name=INDEX_NAME, limit=args.chunk_limit)
    print(f"Loaded {len(chunks)} chunks")

    candidate_chunks = [c for c in chunks if len(c.text.split()) >= args.min_text_length]
    print(f"{len(candidate_chunks)} chunks remain after length filtering")

    if not candidate_chunks:
        raise ValueError("No candidate chunks found after filtering")

    sample_size = min(args.num_samples, len(candidate_chunks))
    sampled_chunks = random.sample(candidate_chunks, sample_size)

    output = []

    for i, chunk in enumerate(sampled_chunks, start=1):
        print(
            f"[{i}/{sample_size}] Generating queries for "
            f"{chunk.show_filename_prefix} / {chunk.episode_filename_prefix}"
        )

        try:
            queries = generate_queries_for_chunk(
                chunk=chunk,
                generator_model=args.generator_model,
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        if not queries:
            print("  No valid queries generated")
            continue

        output.append(
            {
                "show_filename_prefix": chunk.show_filename_prefix,
                "episode_filename_prefix": chunk.episode_filename_prefix,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "text_preview": " ".join(chunk.text.split())[:300],
                "queries": queries,
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    flat_queries = []
    for item in output:
        flat_queries.extend(item["queries"])

    flat_queries = deduplicate_queries(flat_queries)

    simple_query_file = args.output_json.replace(".json", "_flat.json")
    with open(simple_query_file, "w", encoding="utf-8") as f:
        json.dump(flat_queries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed query candidates to: {args.output_json}")
    print(f"Saved flat query list to: {simple_query_file}")
    print(f"Generated {len(flat_queries)} unique queries")
    print("\nInspect the detailed file and manually prune any remaining weak queries.")


if __name__ == "__main__":
    main()