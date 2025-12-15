class ArticleIngestor:
  def __init__(self):
    self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    self.db = pymongo.MongoClient("mongodb://localhost:27017/")["journal"]
  
  def ingest_article(self, article_data):
    """Ingest a single article with embedding generation"""
    
    # Generate embedding from title + summary/content
    text_to_embed = f"{article_data['metadata']['title']}. {article_data['content'].get('summary', article_data['content']['full_text'][:1000])}"
    
    embedding = self.embedder.encode(text_to_embed).tolist()
    
    article_data['embedding'] = embedding
    article_data['created_at'] = datetime.now()
    
    self.db.articles.insert_one(article_data)
    
    return article_data['article_id']
  
  def batch_ingest(self, articles_list):
    """Batch ingestion for initial load"""
    for article in articles_list:
      self.ingest_article(article)
      