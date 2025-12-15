# Complete Installation Guide

This guide walks you through installing all dependencies and setting up the RAG system from scratch.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Install Python](#install-python)
3. [Install MongoDB](#install-mongodb)
4. [Download Project Files](#download-project-files)
5. [Setup Python Environment](#setup-python-environment)
6. [Install Dependencies](#install-dependencies)
7. [Download AI Models](#download-ai-models)
8. [Verify Installation](#verify-installation)

---

## System Requirements

- **OS**: Linux (Ubuntu/Debian), macOS, or Windows (with WSL recommended)
- **RAM**: 16GB minimum
- **Disk**: 15GB free space (10GB for models, 5GB for data)
- **CPU**: Multi-core recommended (models run on CPU)
- **Network**: Internet connection for downloading models

---

## Install Python

### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install Python 3.9+
sudo apt-get install -y python3 python3-pip python3-venv

# Verify installation
python3 --version  # Should show 3.9 or higher
```

### macOS

```bash
# Install using Homebrew
brew install python@3.11

# Verify installation
python3 --version
```

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify in Command Prompt: `python --version`

---

## Install MongoDB

### Ubuntu/Debian

```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Update and install
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
sudo systemctl status mongod
```

### macOS

```bash
# Install using Homebrew
brew tap mongodb/brew
brew install mongodb-community@6.0

# Start MongoDB
brew services start mongodb-community@6.0

# Verify
mongosh  # Should connect to MongoDB
```

### Windows

1. Download MongoDB Community from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Run installer (choose "Complete" setup)
3. Install MongoDB Compass (GUI) when prompted
4. MongoDB will start automatically as a service

**Verify:**
```cmd
mongosh
```

---

## Download Project Files

### Option 1: Git Clone (if available)

```bash
git clone <repository-url>
cd rag-journal
```

### Option 2: Manual Download

1. Download all project files
2. Create project directory structure as shown in README.md
3. Copy files to appropriate locations

### Verify Structure

```bash
ls -la

# You should see:
# config/
# data/
# scripts/
# src/
# requirements.txt
# setup.py
# README.md
```

---

## Setup Python Environment

### Create Virtual Environment

```bash
# Navigate to project directory
cd rag-journal

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Your prompt should now show (venv)
```

### Upgrade pip

```bash
pip install --upgrade pip
```

---

## Install Dependencies

### Install All Packages

```bash
# Install from requirements.txt
pip install -r requirements.txt

# This will install:
# - pymongo (MongoDB driver)
# - torch (PyTorch for models)
# - transformers (HuggingFace models)
# - sentence-transformers (embeddings)
# - and other dependencies

# This may take 5-10 minutes
```

### Install Project

```bash
# Install in editable mode
pip install -e .
```

### Verify Installation

```bash
# Check installed packages
pip list | grep -E "torch|transformers|sentence|pymongo"

# Should show versions for each package
```

---

## Download AI Models

Models will download automatically on first use, but you can pre-download them:

### Pre-download Models (Recommended)

Create `download_models.py`:

```python
#!/usr/bin/env python3
"""Pre-download all required models"""

print("Downloading models... This may take 10-15 minutes.\n")

# Download LLM
print("1/2 Downloading LLM (Qwen2.5-3B, ~6GB)...")
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print("✓ LLM downloaded\n")

# Download embeddings
print("2/2 Downloading embedding model (~400MB)...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("✓ Embeddings downloaded\n")

print("="*80)
print("All models downloaded successfully!")
print("Models are cached in: ~/.cache/huggingface/")
print("="*80)
```

Run it:

```bash
python download_models.py
```

**Expected download sizes:**
- LLM: ~6GB
- Embeddings: ~400MB
- Total: ~6.5GB

**Download time:**
- Fast connection (100Mbps): 10-15 minutes
- Medium connection (20Mbps): 30-40 minutes

---

## Verify Installation

### 1. Check MongoDB

```bash
mongosh --eval "db.version()"

# Should print MongoDB version
```

### 2. Setup Database

```bash
python scripts/setup_database.py

# Expected output:
# ================================================================================
# DATABASE SETUP
# ================================================================================
# 
# Connecting to MongoDB...
# ✓ Database indexes created
# ✓ Database connection successful!
# ...
```

### 3. Test Model Loading

```python
# Run Python interactive shell
python

>>> from src.rag.query_router import QueryRouter
>>> router = QueryRouter()
# Should load without errors (first time will download models)
>>> exit()
```

### 4. Check All Components

```bash
# This should work without errors
python -c "
from rag_journal.database.mongodb_client import MongoDBClient
from rag_journal.embeddings.embedder import ArticleEmbedder
from rag_journal.models.article import Article, ArticleMetadata, ArticleContent
from rag_journal.rag.query_router import QueryRouter
print('✓ All components imported successfully')
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Issue: "Can't connect to MongoDB"

**Solution:**
```bash
# Check if MongoDB is running
sudo systemctl status mongod  # Linux
brew services list | grep mongo  # macOS

# Start MongoDB
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

### Issue: "Out of memory during model download"

**Solution:**
```bash
# Download models one at a time
# First, LLM only:
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

# Then, embeddings:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

### Issue: "Torch not found" or "No module named 'torch'"

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch==2.1.2
```

### Issue: Models downloading to wrong location

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk
export TRANSFORMERS_CACHE=/path/to/large/disk

# Then download models
python download_models.py
```

### Issue: "Permission denied" when installing

**Solution:**
```bash
# Don't use sudo with pip in virtual environment
# If you see permission errors, ensure venv is activated
source venv/bin/activate

# Then install again
pip install -r requirements.txt
```

---

## Alternative: Automated Setup

Use the provided setup script:

```bash
# Make it executable
chmod +x setup_project.sh

# Run it
./setup_project.sh
```

This script will:
1. Check Python and MongoDB
2. Create virtual environment
3. Install all dependencies
4. Setup directory structure
5. Initialize database

---

## Post-Installation

After successful installation:

1. **Add example data:**
   ```bash
   # Example files are included
   ls data/articles/aa_example_001
   ```

2. **Ingest example data:**
   ```bash
   python scripts/ingest_articles.py
   ```

3. **Test queries:**
   ```bash
   python scripts/test_queries.py --interactive
   ```

---

## Getting Models Info

View downloaded models:

```bash
# Linux/macOS
ls -lh ~/.cache/huggingface/hub/

# Windows
dir %USERPROFILE%\.cache\huggingface\hub\
```

---

## Next Steps

✅ Installation complete! Continue to [QUICKSTART.md](QUICKSTART.md) for usage instructions.

For detailed documentation, see [README.md](README.md).

---

## Support

If you encounter issues:

1. Check this troubleshooting section
2. Verify all requirements are met
3. Ensure MongoDB is running
4. Check that virtual environment is activated
5. Review error messages carefully

For persistent issues, please open an issue with:
- Error message
- Your OS and Python version
- Steps to reproduce
