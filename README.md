# DD2477_Search_Project

Code for Information Retrieval project

## Local elasticsearch setup

1. Install docker

2. Run the following:

```bash
curl -fsSL https://elastic.co/start-local | sh
```

After installing the services will be running on the following ports:

- Elasticsearch: http://localhost:9200

- Kibana: http://localhost:5601

Test the connection with:

```bash
cd elastic-start-local/
source .env
curl $ES_LOCAL_URL -H "Authorization: ApiKey ${ES_LOCAL_API_KEY}"
```

3. To stop the services, run:

```bash
cd elastic-start-local
./stop.sh
```

4. To start the services again, run:

```bash
cd elastic-start-local
./start.sh
```

5. Uninstall:

```bash
cd elastic-start-local
./uninstall.sh
```
