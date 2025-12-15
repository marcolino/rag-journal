import pymongo
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from rag_journal.utils.logger import logger
from rag_journal.utils.config import CONFIG


class MongoDBClient:
  """MongoDB client for article storage and retrieval"""
  
  def __init__(self):
    """Initialize MongoDB client"""
    db_config = CONFIG['database']
    self.db_config = db_config

    # Connect to MongoDB
    self.client = pymongo.MongoClient(db_config['mongodb_uri'])
    self.db = self.client[db_config['database_name']]
    self.collection = self.db[db_config['collection_name']]
    
    # Create indexes
    self._create_indexes()
  
  def _create_indexes(self):
    """Create necessary indexes for efficient querying, in needed"""
    specs = [
      ("metadata.author", {}),
      ("metadata.publication_date", {}),
      ("metadata.categories", {}),
      ("article_id", {"unique": True})
    ]
    
    # Convert SON → dict → sorted tuple (hashable)
    existing_keys = {
      tuple(sorted(dict(idx["key"]).items())): idx["name"] 
      for idx in self.collection.list_indexes()
    }

    # Check if text index exists by name
    existing_names = {idx["name"] for idx in self.collection.list_indexes()}

    created = 0
    
    # Create search index if they do not exist
    for field, options in specs:
      key = {field: 1}
      key_tuple = tuple(sorted(key.items()))  # ('article_id', 1)
      
      if key_tuple not in existing_keys:
        self.collection.create_index(field, **options)
        created += 1
    
    # Create text search index if it does not exist
    if "text_search_index" not in existing_names:
      self.collection.create_index([
        ("metadata.title", "text"),
        ("metadata.author", "text"),
        ("content.full_text", "text")
      ], name = "text_search_index")
      created += 1

    if created > 0:
      logger.info(f"✓ creati {created} indici nel database")
    # else:
    #   logger.info("ℹ Tutti gli indici esistono")

  def mongo_clean(self, doc):
    """Convert non-BSON types to serializable"""
    if isinstance(doc, dict):
      return {k: self.mongo_clean(v) for k, v in doc.items()}
    elif isinstance(doc, list):
      return [self.mongo_clean(x) for x in doc]
    elif isinstance(doc, date):
      return datetime.combine(doc, datetime.min.time())
    return doc
  
  def insert_article(self, article_data: Dict[str, Any]) -> str:
    """Insert a single article"""
    try:
      article_data_clean = self.mongo_clean(article_data)
      result = self.collection.insert_one(article_data_clean)
      #logger.info("✓ Articolo inserito")
    except pymongo.errors.DuplicateKeyError:
      logger.info("ℹ L'articolo esiste già (ignorato)")
    except pymongo.errors.OperationFailure as e:
      logger.error(f"✗ Errore MongoDB: {e}")
      raise # Rethrow for caller to handle
    except Exception as e:
      logger.error(f"✗ Errore sconociuto: {e}")
      raise # Rethrow for caller to handle
    return str(result.inserted_id)

  
  def insert_articles_batch(self, articles: List[Dict[str, Any]]) -> List[str]:
    """Insert multiple articles"""
    result = self.collection.insert_many(articles)
    return [str(id) for id in result.inserted_ids]
  
  def find_by_filter(
      self, 
      filter_dict: Dict[str, Any], 
      projection: Optional[Dict[str, int]] = None,
      limit: int = 0) -> List[Dict[str, Any]]:
    """Find articles by filter"""
    cursor = self.collection.find(filter_dict, projection)
    if limit > 0:
      cursor = cursor.limit(limit)
    return list(cursor)
  
  def count_by_filter(self, filter_dict: Dict[str, Any]) -> int:
    """Count articles matching filter"""
    return self.collection.count_documents(filter_dict)
  
  def get_all_articles(
      self, 
      projection: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Get all articles (use with caution on large datasets)"""
    return list(self.collection.find({}, projection))
  
  def article_exists(self, article_id: str) -> bool:
    """Check if article exists"""
    return self.collection.count_documents({"article_id": article_id}) > 0
  
  def update_article(self, article_id: str, update_data: Dict[str, Any]) -> bool:
    """Update an article"""
    result = self.collection.update_one(
      {"article_id": article_id},
      {"$set": update_data}
    )
    return result.modified_count > 0
  
  def delete_article(self, article_id: str) -> bool:
    """Delete an article"""
    result = self.collection.delete_one({"article_id": article_id})
    return result.deleted_count > 0
  
  def clear_collection(self):
    """Clear all articles (use with caution!)"""
    self.collection.delete_many({})
    logger.info("⚠ Tutti gli articoli cancellati dalla collezione")
  
  def get_statistics(self) -> Dict[str, Any]:
    """Get collection statistics"""
    total_articles = self.collection.count_documents({})
    
    # Get unique authors
    authors = self.collection.distinct("metadata.author")
    
    # Get date range
    oldest = self.collection.find_one(
      sort=[("metadata.publication_date", pymongo.ASCENDING)]
    )
    newest = self.collection.find_one(
      sort=[("metadata.publication_date", pymongo.DESCENDING)]
    )
    
    return {
      "total_articles": total_articles,
      "unique_authors": len(authors),
      "date_range": {
        "oldest": oldest['metadata']['publication_date'] if oldest else None,
        "newest": newest['metadata']['publication_date'] if newest else None
      }
    }
  
  def text_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Full-text search using MongoDB text index"""
    return list(self.collection.find(
      {'$text': {'$search': query}},
      {'score': {'$meta': 'textScore'}}
    ).sort([('score', {'$meta': 'textScore'})]).limit(limit))
  