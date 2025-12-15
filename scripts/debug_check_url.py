#!/usr/bin/env python3
"""
Check url value in all articles
"""

from rag_journal.database.mongodb_client import MongoDBClient

db = MongoDBClient()

# Get one article
articles = db.get_all_articles()

if articles:
  print("First article in database:")
  print("="*80)
  import json
  print(json.dumps(articles[0], indent = 2, default = str))
  print("\n" + "="*80)
  print("\nKeys in article:", articles[0].keys())
  print("\nHas 'url' field?", 'url' in articles[0])
  if 'url' in articles[0]:
    print(f"URL value: {articles[0]['url']}")
else:
  print("No articles in database!")