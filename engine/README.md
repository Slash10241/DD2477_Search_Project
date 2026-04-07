# Engine (Django Backend + HTMX frontend)

## Quick Start

Run these commands from the `engine` directory:

```bash
cd engine
uv sync
uv run python manage.py check
uv run python manage.py runserver
```

The server will be available at:

`http://127.0.0.1:8000/`

## Required `.env` for Services

Create this file:

`engine/web/services/.env`

With at least:

```env
API_KEY=your_elasticsearch_api_key
METADATA_TSV_PATH=/absolute/path/to/metadata.tsv
```

Notes:

- `API_KEY` is used for Elasticsearch queries.
- `METADATA_TSV_PATH` is used to load show/episode metadata for result enrichment.
- Make sure Elasticsearch is running (at http://localhost:9200) and the `podcasts` index has been created/populated.
