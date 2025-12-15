from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from rag_journal.utils.config import CONFIG


@dataclass
class ArticleMetadata:
  """Article metadata structure"""
  author: str
  publication_date: datetime
  categories: List[str]
  translator: Optional[str]
  title: str
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for MongoDB"""
    data = asdict(self)
    # Ensure datetime is properly formatted
    if isinstance(data['publication_date'], datetime):
      data['publication_date'] = data['publication_date']
    return data


@dataclass
class ArticleContent:
  """Article content structure"""
  full_text: str
  word_count: int
  summary: Optional[str] = None
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for MongoDB"""
    return asdict(self)


@dataclass
class Article:
  """Complete article structure"""
  article_id: str
  metadata: ArticleMetadata
  content: ArticleContent
  embedding: Optional[List[float]] = None
  created_at: Optional[datetime] = None
  
  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for MongoDB"""
    return {
      "article_id": self.article_id,
      "metadata": self.metadata.to_dict(),
      "content": self.content.to_dict(),
      "embedding": self.embedding,
      "created_at": self.created_at or datetime.now()
    }
  
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'Article':
    """Create Article from MongoDB document"""
    metadata = ArticleMetadata(
      author = data['metadata']['author'],
      publication_date = data['metadata']['publication_date'],
      categories = data['metadata']['categories'],
      translator = data['metadata'].get('translator'),
      title = data['metadata']['title']
    )
    
    content = ArticleContent(
      full_text = data['content']['full_text'],
      word_count = data['content']['word_count'],
      summary = data['content'].get('summary')
    )
    
    return cls(
      article_id = data['article_id'],
      metadata = metadata,
      content = content,
      embedding = data.get('embedding'),
      created_at = data.get('created_at')
    )
  