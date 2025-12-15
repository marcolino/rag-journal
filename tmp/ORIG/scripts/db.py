{
  "_id": ObjectId("..."),
  "article_id": "art_123",
  "metadata": {
    "author": "Mario Rossi",
    "publication_date": ISODate("2023-05-15"),
    "categories": ["politica estera", "ue"],
    "translator": "Giovanni Bianchi",
    "title": "La nuova strategia europea"
  },
  "content": {
    "full_text": "...",
    "word_count": 2500,
    "summary": "..."  // Optional: generated during ingestion
  },
  "embedding": [0.123, -0.456, ...],  // 384 or 768 dimensions
  "created_at": ISODate("...")
}

// Indexes
db.articles.createIndex({"metadata.author": 1})
db.articles.createIndex({"metadata.publication_date": 1})
db.articles.createIndex({"metadata.categories": 1})
db.articles.createIndex({"embedding": "vector"})  // MongoDB Atlas Vector Search