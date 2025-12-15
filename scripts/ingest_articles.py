#!/usr/bin/env python3
"""
Article Ingestion Script

This script ingests articles from the filesystem into MongoDB with embeddings.

Expected file structure:
data/
  articles/
  article_001.txt
  article_002.txt
  ...

YAML article format:
title: "Sample Article"
url: "https://example.com/article"
categories: ["Technology"]
source: "News Site"
publication_date_source: "2025-12-09"
author: "John Doe"
translator: "Jane Smith"
number: 123
content: |-
  word 1, word 2, ...
  word N.
"""

import sys
import yaml
import click
from datetime import datetime
from dateutil import parser
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from rag_journal.database.mongodb_client import MongoDBClient
from rag_journal.embeddings.embedder import ArticleEmbedder
from rag_journal.models.article import Article, ArticleMetadata, ArticleContent


class ArticleIngestor:
  """Handle article ingestion with embedding generation"""
  
  def __init__(self):
    """Initialize ingestor"""
    self.db = MongoDBClient()
    self.embedder = ArticleEmbedder()
  
  def load_article_from_yaml(self, yaml_path: str) -> Dict[str, Any]:
    """Load article from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
      return yaml.safe_load(f)
  
  def create_article_from_yaml(self, article_id: str, yaml_path: str) -> Article:
    """Create Article object from YAML file"""
    
    # Load YAML data
    data = self.load_article_from_yaml(yaml_path)
    
    # Parse publication date (TODO: handle both publication_date_source and publication_date_resistenze ?)
    try:
      #publication_date = datetime.strptime(data['publication_date_source'], '%Y-%m-%d')
      publication_date = fuzzy_parse_date(data['publication_date_source'])
    except (ValueError, KeyError) as e:
      # Fallback to current date if parsing fails
      publication_date = None
      #print(f"  ⚠ Warning: Could not parse publication_date for {article_id}")
      #raise e from None
    
    # Extract categories from topics and category
    categories = []
    if 'categories' in data and data['categories']:
      # Categories can be a list or a string
      if isinstance(data['categories'], list):
        categories.extend([str(t).strip() for t in data['categories'] if t])
      else:
        categories.append(str(data['categories']).strip())
    
    # Remove duplicates and empty strings
    categories = list(set([c for c in categories if c]))
    
    # Create metadata object
    metadata = ArticleMetadata(
      author = data.get('author', '<unknown>'),
      publication_date = publication_date,
      categories = categories,
      translator = data.get('translator'),
      title = data.get('title', ' <untitled>')
    )
    
    # Get content
    content_text = data.get('content', '')
    if not content_text:
      raise ValueError(f"Article {article_id} has no content, ignoring it")
    
    # Create content object
    word_count = len(content_text.split())
    content = ArticleContent(
      full_text = content_text,
      word_count = word_count,
      summary = None # Can be added later if needed
    )
    
    # Create article
    article = Article(
      article_id = article_id,
      metadata = metadata,
      content = content
    )
    
    # Store additional fields as metadata (optional)
    article.url = data.get('url')
    article.source = data.get('source')
    article.number = data.get('number')
    
    return article
  
  def ingest_article(self, article: Article, skip_if_exists: bool = True) -> bool:
    """Ingest single article with embedding"""
    
    # Check if exists
    if skip_if_exists and self.db.article_exists(article.article_id):
      return False
    
    # Generate embedding
    text_to_embed = self.embedder.prepare_article_text(
      article.metadata.title,
      article.content.full_text
    )
    embedding = self.embedder.embed_text(text_to_embed)
    
    # Add embedding to article
    article.embedding = embedding
    article.created_at = datetime.now()
    
    # Convert to dict and add extra fields
    article_dict = article.to_dict()
    if hasattr(article, 'url'):
      article_dict['url'] = article.url
    if hasattr(article, 'source'):
      article_dict['source'] = article.source
    if hasattr(article, 'number'):
      article_dict['number'] = article.number
    
    # Insert to database
    #try:
    result = self.db.insert_article(article_dict)
    return True

  def ingest_batch(
    self, 
    articles_dir: str,
    skip_existing: bool = True
  ) -> Dict[str, int]:
    """Ingest all articles from directory"""
    
    articles_path = Path(articles_dir)
    
    if not articles_path.exists():
      raise ValueError(f"Articles directory not found: {articles_dir}")
    
    # Find all files (no extension filter)
    # Exclude common non-article files
    exclude_patterns = {'.DS_Store', 'README', '.gitkeep', '.gitignore'}
    article_files = [
      f for f in sorted(articles_path.iterdir()) 
      if f.is_file() and f.name not in exclude_patterns and not f.name.startswith('.')
    ]
    
    stats = {
      "total": len(article_files),
      "ingested": 0,
      "skipped": 0,
      "errors": 0
    }
    
    print(f"\nFound {stats['total']} article files in {articles_dir}")
    
    # Process each article
    for article_file in tqdm(article_files, desc = "Ingesting articles"):
      try:
        # Use filename as article ID
        article_id = article_file.name
        
        # Create article from YAML
        article = self.create_article_from_yaml(
          article_id,
          str(article_file)
        )
        
        # Ingest
        ingested = self.ingest_article(article, skip_if_exists = skip_existing)
        
        if ingested:
          stats['ingested'] += 1
        else:
          stats['skipped'] += 1
          
      except yaml.YAMLError as e:
        print(f"\n✗ YAML Error in {article_file.name}: {e}")
        stats['errors'] += 1
      except Exception as e:
        print(f"\n✗ Error processing {article_file.name}: {e}")
        stats['errors'] += 1
    
    return stats

def load_config(ctx, param, value):
  """Eager callback to load config.yaml defaults"""
  config_path = Path("config/config.yaml")
  if config_path.exists():
    with open(config_path) as f:
      config = yaml.safe_load(f)
  articles_path = config.get("ingestion", {}).get("articles_path", "data/articles")
  ctx.default_map = {"articles_dir": articles_path}
  return value  # Allow --config override if added later


def fuzzy_parse_date(date_str):
  try:
    return parser.parse(date_str, fuzzy = True).date()
  except:
    return None


@click.command()
@click.option('--articles-dir', 
              default = 'data/articles', 
              help = 'Directory containing article YAML files (no extension)')
@click.option('--config', is_eager = True, callback = load_config, expose_value = False, hidden = True)

@click.option('--skip-existing/--overwrite', default = True,
        help = 'Skip articles that already exist in DB')
@click.option('--clear-db', is_flag = True,
        help = 'Clear database before ingestion (DANGEROUS!)')

def main(articles_dir, skip_existing, clear_db):
  """Ingest articles from YAML files into MongoDB"""
  
  print("="*80)
  print("ARTICLE INGESTION SYSTEM")
  print("="*80)
  
  # Initialize ingestor
  ingestor = ArticleIngestor()
  
  # Clear database if requested
  if clear_db:
    response = input("⚠ WARNING: This will delete all articles! Type 'YES' to confirm: ")
    if response == "YES":
      ingestor.db.clear_collection()
    else:
      print("Aborted.")
      return
  
  # Ingest articles
  try:
    stats = ingestor.ingest_batch(articles_dir, skip_existing)
    
    # Print statistics
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)
    print(f"Total files: {stats['total']}")
    print(f"✓ Ingested: {stats['ingested']}")
    print(f"⊘ Skipped: {stats['skipped']}")
    print(f"✗ Errors: {stats['errors']}")
    
    # Show database statistics
    db_stats = ingestor.db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"Total articles in DB: {db_stats['total_articles']}")
    print(f"Unique authors: {db_stats['unique_authors']}")
    if db_stats['date_range']['oldest']:
      print(f"Date range: {db_stats['date_range']['oldest'].date()} to {db_stats['date_range']['newest'].date()}")
  
  except Exception as e:
    print(f"\n✗ Fatal error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


if __name__ == "__main__":
  main()
  