from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import os
from pathlib import Path
from rag_journal.utils.logger import logger
from rag_journal.utils.config import CONFIG

class ArticleEmbedder:
  """Handle article embedding generation"""
  
  def __init__(self):
    """Initialize embedder"""
    embedding_config = CONFIG['models']['embedding']
    
    self.model_name = embedding_config['model_name']
    self.batch_size = embedding_config['batch_size']
    self.dimension = embedding_config['dimension']
    
    logger.info(f"Si carica il modello di embedding: {self.model_name}")
    
    # Try to load from cache first
    model_path = self._get_cached_model_path(self.model_name)
    
    try:
      # Try loading with local_files_only
      self.model = SentenceTransformer(model_path, device='cpu')
      logger.info("✓ Modello di embedding caricato alla cache")
    except Exception as e:
      logger.info(f"  Cache non trovata, si scarica il modello...")
      # If not in cache, download (requires internet)
      self.model = SentenceTransformer(self.model_name, device='cpu')
      logger.info("✓ Modello di embedding scaricato e caricato")
  
  def _get_cached_model_path(self, model_name: str) -> str:
    """Get the path to cached model, or return model_name if not cached"""
    cache_dir = os.environ.get('HF_HOME', 
                   os.path.join(Path.home(), '.cache', 'huggingface'))
    
    # Convert model name to cache format
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, 'hub', model_cache_name)
    
    # Check if model exists in cache
    if os.path.exists(model_cache_path):
      snapshots_dir = os.path.join(model_cache_path, 'snapshots')
      if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
          full_path = os.path.join(snapshots_dir, snapshots[0])
          return full_path
    
    # Not in cache - will require download
    return model_name
  
  def embed_text(self, text: str) -> List[float]:
    """Generate embedding for a single text"""
    embedding = self.model.encode(text, convert_to_tensor=False)
    return embedding.tolist()
  
  def embed_batch(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    embeddings = self.model.encode(
      texts, 
      batch_size=self.batch_size,
      show_progress_bar=True,
      convert_to_tensor=False
    )
    return embeddings.tolist()
  
  def prepare_article_text(self, title: str, content: str, max_length: int = 1000) -> str:
    """
    Prepare article text for embedding.
    Combines title with content preview to create meaningful embedding.
    """
    # Use title + first part of content for embedding
    content_preview = content[:max_length] if len(content) > max_length else content
    return f"{title}. {content_preview}"
  
  @staticmethod
  def cosine_similarity(embedding1: Union[List[float], np.ndarray], 
             embedding2: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two embeddings"""
    a = np.array(embedding1)
    b = np.array(embedding2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
  
  @staticmethod
  def rank_by_similarity(query_embedding: List[float], 
              article_embeddings: List[tuple]) -> List[tuple]:
    """
    Rank articles by similarity to query.
    
    Args:
      query_embedding: Query embedding vector
      article_embeddings: List of tuples (article_data, embedding)
    
    Returns:
      Sorted list of tuples (article_data, embedding, similarity_score)
    """
    scored = []
    for article_data, embedding in article_embeddings:
      score = ArticleEmbedder.cosine_similarity(query_embedding, embedding)
      scored.append((article_data, embedding, score))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored
    