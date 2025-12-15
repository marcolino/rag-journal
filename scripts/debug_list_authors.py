#!/usr/bin/env python3
"""
List all authors in articles
"""

from rag_journal.database.mongodb_client import MongoDBClient

db = MongoDBClient()

# Get all articles with metadata projection
articles = db.get_all_articles(projection={"metadata": 1})

# Extract unique authors
all_authors = set()

for article in articles:
  # Use dictionary access, not attribute access
  metadata = article.get('metadata', {})
  
  if metadata:  # Check if metadata dict exists and is not empty
    author = metadata.get('author')
    
    if author:
      # Handle different author formats
      if isinstance(author, str):
        # Split by comma if multiple authors in one string
        authors = [a.strip() for a in author.split(',') if a.strip()]
        all_authors.update(authors)
      elif isinstance(author, list):
        authors = [a.strip() if isinstance(a, str) else str(a) 
                  for a in author if a]
        all_authors.update(authors)

print(f"Found {len(all_authors)} unique authors:")
for author in sorted(all_authors):
  print(f"  - {author}")
