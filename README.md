# Italian Journal RAG System

A Retrieval-Augmented Generation (RAG) system for querying a corpus of Italian journal articles. The system uses local open-source models and respects article-level granularity.

## Features

- **Article-level organization**: Maintains article integrity (no arbitrary chunking)
- **Intelligent query routing**: Automatically classifies queries and routes to appropriate execution path
- **Hybrid search**: Combines metadata filtering with semantic search
- **Local execution**: Runs entirely on CPU with open-source models
- **Italian language support**: Optimized for Italian text

## System Requirements

- Python 3.9+
- MongoDB 4.0+
- 16GB RAM minimum
- ~10GB disk space for models and data

## Installation

### 1. Clone and Setup Project

```bash
# Create project directory
mkdir rag-journal
cd rag-journal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 2. Install and Start MongoDB

**On Ubuntu/Debian:**
```bash
sudo apt-get install mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

**On macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**On Windows:**
Download from [MongoDB Download Center](https://www.mongodb.com/try/download/community) and follow installation instructions.

**Verify MongoDB is running:**
```bash
mongosh  # or mongo
# Should connect without errors
```

### 3. Download Models

Models will be downloaded automatically on first run. This may take several minutes:

- **LLM**: Qwen/Qwen2.5-3B-Instruct (~6GB)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (~400MB)

To pre-download models:

```python
# Run this Python script to pre-cache models
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Download LLM
print("Downloading LLM...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Download embeddings
print("Downloading embedding model...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("✓ All models downloaded!")
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env if needed (optional - defaults should work)
```

### 5. Setup Database

```bash
python scripts/setup_database.py
```

## Data Preparation

### Article Format

Place your articles in `data/articles/` as `YAML` files:

```
data/articles/article_001
data/articles/article_002
...
```

### Metadata
url: https://example.com/article_001.html
title: Per il Centro Studi Vietnamiti e Associazione Italia Vietnam, Torino
categories: [associazione e dintorni]
publication_date_source: '2025-01-01'
number: 123
author: 'Mario Rossi'
source: 'La Stampa'
translator: 'Adele Benatti'
contents: |-
  text ...
  ...

**Required fields:**
- `title`: string
- `author`: string
- `publication_date`: ISO date (YYYY-MM-DD) or null
- `categories`: array of strings or empty array
- `translator`: string or null

**File naming:** File names are completely free

### Example Data

Example files are provided:
- `data/articles/aa_example_001`

## Usage

### 1. Ingest Articles

```bash
# Ingest all articles
python scripts/ingest_articles.py

# With options
python scripts/ingest_articles.py --articles-dir data/articles

# Clear database and re-ingest (CAREFUL!)
python scripts/ingest_articles.py --clear-db

# Overwrite existing articles
python scripts/ingest_articles.py --overwrite
```

**Expected output:**
```
Loading embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
✓ Embedding model loaded
Loading LLM model: Qwen/Qwen2.5-3B-Instruct
✓ LLM model loaded
✓ RAG system ready

Found 20000 article files
Ingesting articles: 100%|██████████| 20000/20000
```

### 2. Query the System

**Interactive mode:**
```bash
python scripts/test_queries.py --interactive
```

**Batch test mode:**
```bash
python scripts/test_queries.py --batch
```

### 3. Use in Your Code

```python
from src.rag.query_router import QueryRouter

# Initialize router
router = QueryRouter()

# Ask a question
result = router.query("Quanti articoli ha scritto Mario Rossi nel 2023?")

print(result['answer'])
# Output: "Ho trovato 15 articoli che corrispondono ai criteri."

# Semantic search
result = router.query("Cosa dice l'autore sull'energia rinnovabile?")
print(result['answer'])
# Output with citations from relevant articles
```

## Query Types

The system automatically handles different query types:

### Metadata Queries
```
"Quanti articoli di Mario Rossi nel 2023?"
"Articoli sulla categoria 'politica estera'"
"Chi ha scritto di più nel 2022?"
```

### Semantic Queries
```
"Quali articoli parlano di energia rinnovabile?"
"Decarbonizzazione e cambiamento climatico"
"La posizione sulla guerra in Ucraina"
```

### Hybrid Queries
```
"Cosa dice Mario Rossi sull'Ucraina?"
"Articoli di Giulia Verdi su energia nel 2023"
```

### Analytical Queries
```
"Qual è il trend della politica energetica dal 2022?"
"Come è cambiata la narrativa sulla NATO?"
```

## Performance

On a machine with 16GB RAM and CPU-only:

- **Ingestion**: ~500 articles/hour (including embedding generation)
- **Query classification**: 2-5 seconds
- **Semantic search**: 1-3 seconds (over 20k articles)
- **Answer generation**: 5-15 seconds

## Architecture

```
User Query
    ↓
[Query Classifier (LLM)]
    ↓
Classification: metadata | semantic | hybrid | analytical
    ↓
[Query Router]
    ↓
├─→ [MongoDB Filter] → Count/List results
├─→ [Semantic Search] → Rank by similarity
└─→ [Hybrid] → Filter + Semantic
    ↓
[Answer Generator (LLM)]
    ↓
Final Answer
```

## Project Structure

```
rag-journal/
├── config/
│   └── config.yaml           # System configuration
├── data/
│   ├── articles/             # Article YAML files
├── src/rag_journal/
│   ├── database/             # MongoDB client
│   ├── embeddings/           # Embedding generation
│   ├── llm/                  # LLM query classifier
│   ├── models/               # Data models
│   └── rag/                  # Query router
└── scripts/
    ├── ingest_articles.py    # Ingestion script
    ├── setup_database.py     # Database setup
    └── test_queries.py       # Query testing
```

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
sudo systemctl status mongodb

# Start MongoDB
sudo systemctl start mongodb

# Check logs
sudo tail -f /var/log/mongodb/mongod.log
```

### Out of Memory

If you encounter OOM errors:

1. Reduce batch size in `config/config.yaml`:
```yaml
ingestion:
  batch_size: 5  # Reduce from 10
```

2. Use quantized models (edit `src/llm/query_classifier.py`):
```python
# Add load_in_4bit=True
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_name,
    torch_dtype=torch.float32,
    device_map=self.device,
    load_in_4bit=True  # Add this
)
```

### Slow Query Performance

1. Ensure indexes are created:
```bash
python scripts/setup_database.py
```

2. For very large datasets (>50k articles), consider:
   - Using MongoDB Atlas with vector search
   - Implementing pagination
   - Pre-filtering by date ranges

### Model Download Issues

If model downloads fail:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk

# Download with retry
pip install huggingface_hub[cli]
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

## Customization

### Using Different Models

Edit `config/config.yaml`:

```yaml
models:
  llm:
    model_name: "microsoft/Phi-3-mini-4k-instruct"  # Alternative LLM
  embedding:
    model_name: "sentence-transformers/distiluse-base-multilingual-cased-v2"
```

### Adjusting Retrieval

```yaml
retrieval:
  max_results: 10              # Return more articles
  min_similarity_score: 0.2    # Lower threshold for more results
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues and questions, please open an issue on the GitHub repository.
