#!/usr/bin/env bash

# Complete Project Setup Script
# This script sets up the entire RAG system from scratch

set -e  # Exit on error

echo "================================================================================"
echo "ITALIAN JOURNAL RAG SYSTEM - SETUP"
echo "================================================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if printf '%s\n%s\n' "$python_version" "$required_version" | sort -V -C; then
  echo -e "${RED}✗ Python 3.9+ required. Found: $python_version${NC}"
  exit 1
else
  echo -e "${GREEN}✓ Python version OK: $python_version${NC}"
fi

# Check MongoDB
echo -e "\n${YELLOW}Checking MongoDB...${NC}"
if command -v mongosh &> /dev/null || command -v mongo &> /dev/null; then
  echo -e "${GREEN}✓ MongoDB client found${NC}"
  
  # Try to connect
  if mongosh --eval "db.version()" &> /dev/null || mongo --eval "db.version()" &> /dev/null; then
    echo -e "${GREEN}✓ MongoDB is running${NC}"
  else
    echo -e "${YELLOW}⚠ MongoDB client found but not running${NC}"
    echo -e "  Start MongoDB with: sudo systemctl start mongodb"
    echo -e "  Or on macOS: brew services start mongodb-community"
  fi
else
  echo -e "${RED}✗ MongoDB not found${NC}"
  echo -e "  Install with:"
  echo -e "  Ubuntu/Debian: sudo apt-get install mongodb"
  echo -e "  macOS: brew install mongodb-community"
  exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo -e "${GREEN}✓ Virtual environment created${NC}"
else
  echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
echo -e "  This may take several minutes..."

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --quiet

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install project
echo -e "\n${YELLOW}Installing project...${NC}"
#pip install -e . --quiet
pip install . --quiet
echo -e "${GREEN}✓ Project installed${NC}"

# Create directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p config
mkdir -p src/rag_journal/{database,embeddings,llm,models,rag,ingestion}
mkdir -p scripts
mkdir -p data/articles
mkdir -p tests

# Create __init__.py files
touch src/rag_journal/__init__.py
touch src/rag_journal/database/__init__.py
touch src/rag_journal/embeddings/__init__.py
touch src/rag_journal/llm/__init__.py
touch src/rag_journal/models/__init__.py
touch src/rag_journal/rag/__init__.py
touch src/rag_journal/ingestion/__init__.py
touch tests/__init__.py

echo -e "${GREEN}✓ Directory structure created${NC}"

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
  fi
fi

# Setup database
echo -e "\n${YELLOW}Setting up database...${NC}"
if python scripts/setup_database.py; then
  echo -e "${GREEN}✓ Database setup complete${NC}"
else
  echo -e "${YELLOW}⚠ Database setup had issues (this is OK if MongoDB isn't running yet)${NC}"
fi

# Summary
echo -e "\n================================================================================"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "================================================================================"
echo -e "\nNext steps:"
echo -e "  1. Place your articles with metadata, in YAML format, in: ${YELLOW}data/articles/${NC}"
echo -e "  2. Ingest articles: ${YELLOW}python scripts/ingest_articles.py${NC}"
echo -e "  3. Test queries: ${YELLOW}python scripts/test_queries.py --interactive${NC}"
echo -e "\nUseful commands:"
echo -e "  - Activate environment: ${YELLOW}source .venv/bin/activate${NC}"
echo -e "  - Check database: ${YELLOW}python scripts/setup_database.py${NC}"
echo -e "\nDocumentation:"
echo -e "  - Full guide: README.md"
echo -e "  - Quick start: QUICKSTART.md"
echo -e "\n${GREEN}Happy querying! 🚀${NC}\n"
