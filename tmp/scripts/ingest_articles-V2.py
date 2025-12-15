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
category: "Technology"
topics: [topic 1, topic 2]
date_publishing: "2025-12-09"
number: 123
author: "John Doe"
source: "News Site"
translator: "Jane Smith"
content: |-
  word 1, word 2, ...
  word N.
"""

#import os
import sys
import json
import click
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from rag_journal.database.mongodb_client import MongoDBClient
from rag_journal.embeddings.embedder import ArticleEmbedder
from rag_journal.models.article import Article, ArticleMetadata, ArticleContent


class ArticleIngestor:
  """Handle article ingestion with embedding generation"""
  
  def __init__(self, config_path: str = "config/config.yaml"):
    """Initialize ingestor"""
    self.db = MongoDBClient(config_path)
    self.embedder = ArticleEmbedder(config_path)
  
  def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file"""
    with open(metadata_path, 'r', encoding = 'utf-8') as f:
      return json.load(f)
  
  def load_article_text(self, article_path: str) -> str:
    """Load article text from file"""
    with open(article_path, 'r', encoding = 'utf-8') as f:
      return f.read()
  
  def create_article_from_files(self, 
                  article_id: str,
                  article_path: str, 
                  metadata_path: str) -> Article:
    """Create Article object from files"""
    
    # Load data
    full_text = self.load_article_text(article_path)
    metadata_dict = self.load_metadata(metadata_path)
    
    # Create metadata object
    metadata = ArticleMetadata(
      author = metadata_dict['author'],
      publication_date = datetime.fromisoformat(metadata_dict['publication_date']),
      categories = metadata_dict['categories'],
      translator = metadata_dict.get('translator'),
      title = metadata_dict['title']
    )
    
    # Create content object
    word_count = len(full_text.split())
    content = ArticleContent(
      full_text = full_text,
      word_count = word_count,
      summary = metadata_dict.get('summary')  # Optional pre-computed summary
    )
    
    # Create article
    article = Article(
      article_id = article_id,
      metadata = metadata,
      content = content
    )
    
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
    
    # Insert to database
    self.db.insert_article(article.to_dict())
    
    return True
  
  def ingest_batch(self, 
          articles_dir: str, 
          metadata_dir: str,
          skip_existing: bool = True) -> Dict[str, int]:
    """Ingest all articles from directories"""
    
    articles_path = Path(articles_dir)
    metadata_path = Path(metadata_dir)
    
    # Find all article files
    article_files = sorted(articles_path.glob("*"))
    
    stats = {
      "total": len(article_files),
      "ingested": 0,
      "skipped": 0,
      "errors": 0
    }
    
    print(f"\nFound {stats['total']} article files")
    
    # Process each article
    for article_file in tqdm(article_files, desc = "Ingesting articles"):
      try:
        # Derive article ID from filename
        article_id = article_file.stem  # filename without extension
        
        # Find corresponding metadata file
        metadata_file = metadata_path / f"{article_id}.json"
        
        if not metadata_file.exists():
          print(f"\n⚠ Warning: No metadata found for {article_id}")
          stats['errors'] += 1
          continue
        
        # Create article
        article = self.create_article_from_files(
          article_id,
          str(article_file),
          str(metadata_file)
        )
        
        # Ingest
        ingested = self.ingest_article(article, skip_if_exists=skip_existing)
        
        if ingested:
          stats['ingested'] += 1
        else:
          stats['skipped'] += 1
          
      except Exception as e:
        print(f"\n✗ Error processing {article_file.name}: {e}")
        stats['errors'] += 1
    
    return stats


@click.command()
@click.option('--articles-dir', default='data/articles', 
        help = 'Directory containing article text files')
@click.option('--metadata-dir', default='data/metadata',
        help = 'Directory containing metadata JSON files')
@click.option('--config', default='config/config.yaml',
        help = 'Config file path')
@click.option('--skip-existing/--overwrite', default=True,
        help = 'Skip articles that already exist in DB')
@click.option('--clear-db', is_flag=True,
        help = 'Clear database before ingestion (DANGEROUS!)')
def main(articles_dir, metadata_dir, config, skip_existing, clear_db):
  """Ingest articles from filesystem into MongoDB"""
  
  print("="*80)
  print("ARTICLE INGESTION SYSTEM")
  print("="*80)
  
  # Initialize ingestor
  ingestor = ArticleIngestor(config)
  
  # Clear database if requested
  if clear_db:
    response = input("⚠ WARNING: This will delete all articles! Type 'YES' to confirm: ")
    if response == "YES":
      ingestor.db.clear_collection()
    else:
      print("Aborted.")
      return
  
  # Ingest articles
  stats = ingestor.ingest_batch(articles_dir, metadata_dir, skip_existing)
  
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


if __name__ == "__main__":
  main()
  