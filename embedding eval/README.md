# Embedding + Reranking Evaluation for Podcast Search

This code evaluates different embedding models for the podcast search project.

The script loads already indexed podcast transcript chunks from the local Elasticsearch index, retrieves the top-k clips using embedding cosine similarity, and then uses an Ollama judge model to score the relevance of the retrieved clips.

The purpose is to help choose a good embedding model for the retriever in the podcast search system.

## What this does

* loads podcast chunks from the local Elasticsearch index `podcasts`
* embeds the chunks using one or more embedding models
* embeds each query and retrieves the top-k most similar chunks
* uses Ollama with `gpt-oss:120b-cloud` as a judge model
* scores each retrieved `(query, chunk)` pair for relevance
* computes retrieval metrics such as:

  * Precision@k
  * MRR
  * nDCG@k
* saves:

  * a JSON file with full results
  * a TXT file with a readable summary

## Requirements

Before running this script, make sure that:

* Elasticsearch is already running locally
* the podcast transcripts have already been indexed into the `podcasts` index
* Ollama is installed
* you are signed in to Ollama so the cloud model `gpt-oss:120b-cloud` can be used
* the required Python dependencies are installed

## Ollama

Install Ollama from the official website, then sign in so the cloud judge model can be used.

After signing in, the script can use:

```text
gpt-oss:120b-cloud
```

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

You can override this by passing a JSON file with queries using `--queries-json`.

Example format:

```json
[
  "Spiritual competition",
  "NBA playoff positioning",
  "Podcast monetization strategies"
]
```

## Example command used for evaluation

```bash
uv run run_embedding_llm_evaluation.py \
  --api-key $ES_LOCAL_API_KEY \
  --queries-json generated_eval_queries_flat.json \
  --chunk-limit 3000 \
  --top-k 10 \
  --judge-model gpt-oss:120b-cloud \
  --models \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "sentence-transformers/all-mpnet-base-v2" \
    "BAAI/bge-small-en-v1.5" \
  --output-json embedding_eval_3models.json \
  --output-txt embedding_eval_3models.txt
```

## Results

```
==============================================================================================================
EMBEDDING MODEL EVALUATION SUMMARY
==============================================================================================================
Model                                                    P@10      MRR    nDCG@10    EmbedTime
--------------------------------------------------------------------------------------------------------------
sentence-transformers/all-mpnet-base-v2              0.6460 0.8880 0.8490     9m 31.5s
BAAI/bge-small-en-v1.5                               0.5490 0.7960 0.7660      3m 6.7s
sentence-transformers/all-MiniLM-L6-v2               0.5410 0.7790 0.7640       42.65s
==============================================================================================================
```

## Notes

* `--chunk-limit 3000` means that the script only loads the first 3000 already indexed chunks from Elasticsearch for evaluation
* this does **not** re-index the dataset
* retrieval is done by embedding the query and the chunks, then ranking chunks by cosine similarity
* the judge scores each retrieved `(query, chunk)` pair multiple times and averages the result

### About EmbedTime

* `EmbedTime` is the **total time required to compute embeddings for all chunks** for a given model
* it is **not an average**
* it reflects the **cost of encoding the full dataset once**

In practice:

* this cost is usually paid once and cached
* it does not affect query-time latency in a deployed system

## Output

The script saves:

* a JSON file with the detailed retrievals, scores, assigned relevance labels, and queries
* a TXT file with a readable summary of the results

---

## Summary

* `all-mpnet-base-v2` gives the best overall retrieval quality but is the slowest
* `bge-small-en-v1.5` provides a good balance between performance and speed
* `all-MiniLM-L6-v2` is the fastest but slightly weaker in retrieval quality

This trade-off helps guide the choice of embedding model depending on system constraints.
