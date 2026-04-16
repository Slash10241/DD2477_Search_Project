# Engine (Django Backend + HTMX frontend)

## Quick Start

Run these commands from the `engine` directory:

```bash
cd engine
uv sync
uv run python manage.py check
uv run python manage.py runserver
````

The server will be available at:

`http://127.0.0.1:8000/`

---

## Required `.env` for Services

Create this file:

`engine/web/services/.env`

With:

```env
API_KEY=your_elasticsearch_api_key
METADATA_TSV_PATH=/absolute/path/to/metadata.tsv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
```

### Notes

* `API_KEY` → used for Elasticsearch queries
* `METADATA_TSV_PATH` → used for loading show/episode metadata
* `GEMINI_API_KEY` → used for LLM features (highlighting + summarization)
* `GEMINI_MODEL` → shared model used across features (not hardcoded)

You can generate a Gemini API key from:
[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Make sure:

* Elasticsearch is running at `http://localhost:9200`
* The `podcasts` index is already created and populated
