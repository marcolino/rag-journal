# Italian Journal RAG System - Project Structure

```
PROJECT/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── config/
│   └── config.yaml
├── src/rag_journal
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── article.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── mongodb_client.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── query_classifier.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── query_router.py
│   └── ingestion/
│       ├── __init__.py
│       └── article_ingestor.py
├── scripts/
│   ├── ingest_articles.py
│   ├── setup_database.py
│   ├── ...
│   └── test_queries.py
├── data/
│   └── articles/          # Articles go here (YAML format, with metadata)
│       ├── article_001
│       ├── article_002
│       └── ...
└── tests/
    ├── __init__.py
    └── test_rag_journal.py
```