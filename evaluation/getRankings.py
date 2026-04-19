"""
Run predefined queries against the podcast search index and save results to a txt file.
Uses hybrid_search for best quality, enriches with metadata, and formats output.
"""

from search.hybrid_search import hybrid_search
from search.metadata_lookup import enrich_results_with_metadata, warm_metadata_cache

# ── Configuration ────────────────────────────────────────────────────────────

MAX_RESULTS = 50
NUM_CANDIDATES = 200  # candidates to consider for kNN; must be >= MAX_RESULTS
K = 50                # nearest neighbours returned by kNN

OUTPUT_FILE = "query_results.txt"

QUERIES = [
    # News & Politics (10)
    "COVID-19 pandemic response and lockdown measures",
    "2020 US presidential election and voter turnout",
    "Black Lives Matter protests and racial justice",
    "Brexit negotiations and the UK leaving the EU",
    "Trump impeachment trial and Senate acquittal",
    "wildfires in Australia and California 2020",
    "Hong Kong protests and China national security law",
    "refugee crisis and immigration policy debates",
    "rise of populism and nationalist movements in Europe",
    "United Nations climate summit and global cooperation",

    # Science & Technology (10)
    "artificial intelligence and machine learning breakthroughs",
    "mRNA vaccines and the future of medicine",
    "climate change policy and the Paris Agreement",
    "SpaceX Falcon 9 and commercial space travel",
    "5G technology rollout and conspiracy theories",
    "CRISPR gene editing and biotech innovation",
    "quantum computing and its practical applications",
    "deepfake technology and synthetic media risks",
    "privacy concerns and surveillance capitalism",
    "autonomous self-driving cars and the future of transport",

    # Business & Economics (10)
    "remote work and the future of the office",
    "stock market crash and economic recession 2020",
    "gig economy and worker rights",
    "startup fundraising and venture capital",
    "Amazon and big tech monopoly concerns",
    "personal finance and investing for beginners",
    "cryptocurrency Bitcoin surge and institutional adoption",
    "supply chain disruption and global trade",
    "universal basic income and wealth inequality",
    "women in leadership and the gender pay gap",

    # Health & Wellbeing (10)
    "mental health during quarantine and isolation",
    "sleep science and improving sleep quality",
    "mindfulness meditation and stress reduction",
    "addiction recovery and substance abuse",
    "diet culture and body positivity movement",
    "long COVID symptoms and post-viral fatigue",
    "therapy and the stigma around seeking help",
    "exercise science and high intensity interval training",
    "gut health microbiome and its effect on mood",
    "burnout and chronic workplace stress",

    # Culture & Society (10)
    "true crime investigations and criminal psychology",
    "social media addiction and its effects on teenagers",
    "diversity and inclusion in the workplace",
    "cancel culture and free speech debate",
    "parenting advice and raising children",
    "feminism and gender equality in 2020",
    "religion spirituality and finding meaning in life",
    "loneliness epidemic and the decline of community",
    "true history of slavery and its lasting legacy",
    "generational differences between millennials and Gen Z",

    # Sport & Entertainment (10)
    "NBA bubble season and LeBron James",
    "history of hip hop and rap music evolution",
    "Tour de France cycling and doping controversies",
    "NFL quarterback rivalries and Super Bowl predictions",
    "Olympics postponement and athlete mental health",
    "Netflix binge watching and the streaming wars",
    "esports gaming industry growth and professional players",
    "Hollywood diversity Oscars so white debate",
    "stand up comedy and the art of the punchline",
    "football soccer tactics and Premier League analysis",
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


def run_all_queries() -> None:
    print("Warming metadata cache …")
    cached = warm_metadata_cache()
    print(f"  {cached} shows loaded.\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for query_text in QUERIES:
            print(f"Querying: {query_text!r}")

            raw_results = hybrid_search(
                query_text=query_text,
                num_candidates=NUM_CANDIDATES,
                k=K,
                max_results=MAX_RESULTS,
            )

            enriched = enrich_results_with_metadata(raw_results)
            num_found = len(enriched)

            # ── Write query header ──────────────────────────────────────────
            fh.write(f"{query_text}\n")
            fh.write(f"{num_found}\n")

            for rank, result in enumerate(enriched, start=1):
                fh.write(format_result(rank, result))

            fh.write("\n")  # blank line between queries
            print(f"  → {num_found} results written.\n")

    print(f"Done. Results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_all_queries()