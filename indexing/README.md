# Indexing the podcast dataset

1. Download the dataset and unzip it into `podcasts-no-audio-13GB`.

2. Unzip the transcripts into a folder:

```bash
cd podcasts-no-audio-13GB/
mkdir ../podcasts-dataset
tar -xzf podcasts-transcripts-0to2.tar.gz -C ../podcasts-dataset
tar -xzf podcasts-transcripts-3to5.tar.gz -C ../podcasts-dataset
tar -xzf podcasts-transcripts-6to7.tar.gz -C ../podcasts-dataset
```

3. Verify that there are 105360 files:

```bash
cd ../podcasts-dataset
find spotify-podcasts-2020/podcasts-transcripts/ -name '*.json' |wc -l
```

4. Run the indexing script:

```bash
uv run index.py \
	--transcripts-dir /path/to/spotify-podcasts-2020/podcasts-transcripts/ \
	--api-key ES_LOCAL_API_KEY
```

The `ES_LOCAL_API_KEY` can be found in the `.env` file in the `elastic-start-local` directory. An optional `--max-files` flag can be used to limit the number of files to process (e.g. for testing).

5. Check that the index exists in Kibana -> Data Management:

Go to `http://localhost:5601` and login with username `elastic` and the password from the `.env` file (`ES_LOCAL_PASSWORD`).

## Embedding model benchmarks

To benchmark the embedding generation performance of different models, run:

```bash
# To run "google/embeddinggemma-300m", accept terms at https://huggingface.co/google/embeddinggemma-300m
hf auth login # Some models like "google/embeddinggemma-300m" require authentication
uv run benchmark_embedding.py --device [auto|cpu|cuda]
```

`--device auto` will use GPU if available, otherwise CPU. Texts and number of runs can be configured in `benchmark_embedding.py`.

To test similarity:

similarity_eval.py  ## test similarity on 2min chunks
------------------
Reads encoder_test_dataset.csv, encodes all texts with each model,
then computes intra-group and inter-group cosine similarities.

Good encoder: high intra-group score, low inter-group score,
              large separation (= intra − inter).

Usage:
    python similarity_eval.py [--device auto|cpu|cuda] [--csv PATH] [--output PATH]

synonym_similarity_eval.py   ## test similarity on synonyms
--------------------------
Reads synonym_test_dataset.csv, embeds each word with each model,
then computes intra-group and inter-group cosine similarities.

Single words are much harder than sentences — models that rely on
contextual cues will behave very differently here than on paragraphs.

Usage:
    python synonym_similarity_eval.py [--device auto|cpu|cuda] [--csv PATH] [--output PATH]

Notes:
    Some models expect a prefix (e.g. nomic uses "search_query: " or
    "search_document: "). Toggle PREFIX_MAP below to experiment.