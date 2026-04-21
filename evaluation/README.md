**Ranking**
Use script getRankings.py to get rankings for each predefined list of queries.
Queries used:
QUERIES = 
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

Results are saved in query_<mode>_results.txt

**Annotation**
LLM Model used to annotate:
gemini-3.1-flash-lite-preview api

Prompt used to annotate the results:
canbe found in getRankingswithRel.py

Annotated file: annotated_<mode>_results.txt

**Evaluate metrics**
Run file evaluate_metrics.py
Results stored in results_<mode> folder.
Results comparision is stored in results_comparision folder.
Had to reduce to just 20 queries and 20 result per query due to api rate limit and using Claude the relevance was inconsistent upon manual check.

**Evaluate LLM Highlighting**
Use the files and steps below for highlight-quality evaluation only.

Input files:

- query_results.txt
- annotated_highlight_quality.txt

Gold annotation format (annotated_highlight_quality.txt):
<query>
<Show name>
<episode name>
<quality_score 0-3>
<\n>

Rules:

- Query line must exactly match query text in query_results.txt
- Keep same result order as query_results.txt
- Score meaning:
  - 0: bad highlight quality
  - 1: weak highlight quality
  - 2: good highlight quality
  - 3: excellent highlight quality

Prompt for annotation:

Model: Claude Sonnet 4.6

- highlight_annotation_prompt.txt

Run highlight evaluation:
uv run evaluate_highlights.py

The script also exports an annotation helper file:

- results/highlight_annotation_input.txt

Use that file as the source when producing annotated_highlight_quality.txt.

Reuse cached predictions (no new LLM calls):
uv run evaluate_highlights.py --reuse-predictions

Highlight outputs in results folder:

- highlight_predictions.json
- highlight_annotation_input.txt
- highlight_metrics_per_query.csv
- highlight_metrics_averaged.csv
- highlight_pr_curve_per_query.csv
- highlight_pr_curve.png
- highlight_pr_curves_per_query/
