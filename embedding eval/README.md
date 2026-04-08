
# Embedding + Reranking Evaluation for Podcast Search

This code evaluates different embedding models for the podcast search project.

The script loads already indexed podcast transcript chunks from the local Elasticsearch index, retrieves the top-k clips using embedding cosine similarity, and then uses an Ollama judge model to score the relevance of the retrieved clips.

The purpose is to help choose a good embedding model for the retriever in the podcast search system.

## What this does

- loads podcast chunks from the local Elasticsearch index `podcasts`
- embeds the chunks using one or more embedding models
- embeds each query and retrieves the top-k most similar chunks
- uses Ollama with `gpt-oss:120b-cloud` as a judge model
- scores each retrieved `(query, chunk)` pair for relevance
- computes retrieval metrics such as:
  - Precision@k
  - MRR
  - nDCG@k
- saves:
  - a JSON file with full results
  - a TXT file with a readable summary

## Requirements

Before running this script, make sure that:

- Elasticsearch is already running locally
- the podcast transcripts have already been indexed into the `podcasts` index
- Ollama is installed
- you are signed in to Ollama so the cloud model `gpt-oss:120b-cloud` can be used
- the required Python dependencies are installed

## Ollama

Install Ollama from the official website, then sign in so the cloud judge model can be used.

After signing in, the script can use:

```text
gpt-oss:120b-cloud
````

as the judge model.

## Queries

The default queries are set in `evaluate_embeddings_llm_judge.py` in:

```python
DEFAULT_QUERIES = [
    "Higgs Boson",
    "terrorism",
    "what Jesus means to me"
]
```

So if you want to change the default queries, edit that list in `evaluate_embeddings_llm_judge.py`.

You can also pass a separate JSON file with queries using `--queries-json`.

Example format for a queries file:

```json
[
  "Higgs Boson",
  "terrorism",
  "what Jesus means to me"
]
```

## Example command

```bash
uv run run_embedding_llm_evaluation.py \
  --api-key $ES_LOCAL_API_KEY \
  --chunk-limit 3000 \
  --top-k 10 \
  --judge-model gpt-oss:120b-cloud \
  --models "sentence-transformers/all-MiniLM-L6-v2" "BAAI/bge-small-en-v1.5" \
  --output-json embedding_eval_small.json \
  --output-txt embedding_eval_small.txt
```

## Notes

* `--chunk-limit 3000` means that the script only loads the first 3000 already indexed chunks from Elasticsearch for evaluation
* this does **not** re-index the dataset
* retrieval is done by embedding the query and the chunks, then ranking chunks by cosine similarity
* the judge scores each retrieved `(query, chunk)` pair multiple times and averages the result

## Output

The script saves:

* a JSON file with the detailed retrievals, scores, assigned relevance labels, and queries
* a TXT file with a readable summary of the results

