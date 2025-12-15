# Quick Start Guide

Get the RAG system running in 10 minutes.

## Prerequisites

- Python 3.9+
- MongoDB installed and running
- 16GB RAM

## Step-by-Step Setup

### 1. Install MongoDB (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y mongodb
sudo systemctl start mongodb
```

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Verify it's running:**
```bash
mongosh  # Should connect without errors
```

### 2. Setup Python Environment

```bash
# Create project directory
mkdir rag-journal
cd rag-journal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create requirements.txt (copy content from the artifacts)
# Then install
pip install -r requirements.txt
```

### 3. Create Project Structure

```bash
# Create directories
mkdir -p config src/{database,embeddings,llm,models,rag,ingestion} scripts data/articles, tests

# Create __init__.py files
touch src/__init__.py
touch src/database/__init__.py
touch src/embeddings/__init__.py
touch src/llm/__init__.py
touch src/models/__init__.py
touch src/rag/__init__.py
touch src/ingestion/__init__.py
```

### 4. Copy All Files

Copy the content from all the provided artifacts into their respective files:

- `config/config.yaml`
- `src/models/article.py`
- `src/database/mongodb_client.py`
- `src/embeddings/embedder.py`
- `src/llm/query_classifier.py`
- `src/rag/query_router.py`
- `scripts/ingest_articles.py`
- `scripts/setup_database.py`
- `scripts/test_queries.py`

### 5. Setup Configuration

```bash
cp .env.example .env
# Edit .env if needed (optional)
```

### 6. Prepare Sample Data

Create example article:

**data/articles/aa_example_001:**
```
url: https://example.com/article_001.html
title: Per il Centro Studi Vietnamiti e Associazione Italia Vietnam, Torino
categories: [associazione e dintorni]
publication_date_source: '2024-01-01'
number: 123
author: 'Mario Rossi'
source: 'La Stampa'
translator: 'Adele Benatti'
contents: |-
La politica energetica europea sta attraversando una fase di trasformazione...
(your article text here)
```

### 7. Initialize Database

```bash
python scripts/setup_database.py
```

Expected output:
```
================================================================================
DATABASE SETUP
================================================================================

Connecting to MongoDB...
✓ Database indexes created
✓ Database connection successful!

Current Statistics:
  Total articles: 0
  Unique authors: 0
  Date range: No articles yet

✓ Database is ready for use!
```

### 8. Ingest Articles

```bash
python scripts/ingest_articles.py
```

**First run will download models** (~6GB, takes 5-10 minutes):
```
Loading embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
✓ Embedding model loaded
Loading LLM model: Qwen/Qwen2.5-3B-Instruct
⚠ This may take a few minutes on first run...
✓ LLM model loaded

Found 1 article files
Ingesting articles: 100%|███████████████████| 1/1

================================================================================
INGESTION COMPLETE
================================================================================
Total files: 1
✓ Ingested: 1
⊘ Skipped: 0
✗ Errors: 0

Database Statistics:
Total articles in DB: 1
Unique authors: 1
Date range: 2023-05-15 to 2023-05-15
```

### 9. Test Queries

```bash
python scripts/test_queries.py --interactive
```

Try these queries:
```
Quanti articoli ci sono nel database?
Cosa dice Mario Rossi sull'energia?
Articoli sull'Europa
```

## Adding Your Articles

### Bulk Import

1. Place all articles in `data/articles/` (YAML format for metadata)
3. Run: `python scripts/ingest_articles.py`

### Creating Metadata from CSV (deprecated)

If you have a CSV with metadata, create a conversion script:

```python
import csv
import json

with open('articles_metadata.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata = {
            "author": row['author'],
            "publication_date": row['date'],  # Format: YYYY-MM-DD
            "categories": row['categories'].split(','),
            "translator": row.get('translator'),
            "title": row['title']
        }
        
        article_id = row['article_id']
        with open(f'data/metadata/{article_id}.json', 'w') as out:
            json.dump(metadata, out, indent=2, ensure_ascii=False)
```

## Next Steps

1. **Add more articles**: Place them in `data/articles/`
2. **Customize models**: Edit `config/config.yaml` to use different models
3. **Integrate into application**: Import `QueryRouter` in your code
4. **Scale up**: For >50k articles, consider MongoDB Atlas with vector search

## Common Issues

**"ModuleNotFoundError":**
```bash
pip install -e .  # Install project in editable mode
```

**"Can't connect to MongoDB":**
```bash
sudo systemctl start mongodb  # Start MongoDB service
```

**"Out of memory":**
- Close other applications
- Reduce `batch_size` in config.yaml
- Consider using quantized models

**Models downloading slowly:**
- Be patient, first download takes time
- Check internet connection
- Models are cached after first download

## Performance Tips

- **First query is slow** (~30 seconds) - models load into memory
- **Subsequent queries are fast** (~5-10 seconds)
- **Pre-load models** on server startup to avoid first-query delay
- **Use SSD** for better model loading times

## Getting Help

- Check the full README.md for detailed documentation
- Review error messages carefully
- Ensure MongoDB is running
- Verify Python version (3.9+)

Happy querying! 🚀
