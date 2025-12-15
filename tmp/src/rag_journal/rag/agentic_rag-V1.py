# import sys
# print(f"Importing agentic_rag. Current module: {__name__}")
# print(f"Current path: {sys.path}")
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List
#from pathlib import Path

from rag_journal.database.mongodb_client import MongoDBClient
from rag_journal.embeddings.embedder import ArticleEmbedder
from rag_journal.embeddings.embedder import ArticleEmbedder
from rag_journal.llm.llm_client import create_llm_client


class AgenticRAG:
  """Agentic RAG system - LLM decides everything"""
  
  def __init__(self, config_path: str = "config/config.yaml"):
    """Initialize agentic RAG"""
    with open(config_path, 'r') as f:
      self.config = yaml.safe_load(f)
    
    print("Initializing Agentic RAG system...")
    
    self.db = MongoDBClient() #config_path)
    self.embedder = ArticleEmbedder() #config_path)
    self.llm = create_llm_client(self.config) #config_path)
    
    self.max_results = self.config['retrieval']['max_results']
    self.semantic_top_k = self.config['retrieval']['semantic_search_top_k']
    
    # Define available tools
    self.tools = self._define_tools()
    
    print("✓ Agentic RAG ready")
  
  def _define_tools(self) -> List[Dict]:
    """Define tools available to the LLM agent"""
    return [
      {
        "name": "search_by_content",
        "description": "Cerca articoli per contenuto semantico. Usa questo quando la domanda riguarda COSA dicono gli articoli su un argomento specifico.",
        "parameters": {
          "query": {
            "type": "string",
            "description": "Query di ricerca semantica (es: 'guerra Ucraina', 'politica energetica')"
          }
        }
      },
      {
        "name": "search_by_author",
        "description": "Trova tutti gli articoli scritti da un autore specifico.",
        "parameters": {
          "author": {
            "type": "string",
            "description": "Nome dell'autore da cercare (es: 'Karl Marx')"
          }
        }
      },
      {
        "name": "search_by_metadata",
        "description": "Filtra articoli usando metadati (autore, data, categorie). Usa questo per query complesse con più filtri.",
        "parameters": {
          "author": {
            "type": "string",
            "description": "Nome autore (opzionale)",
            "optional": True
          },
          "date_from": {
            "type": "string",
            "description": "Data inizio nel formato YYYY-MM-DD (opzionale)",
            "optional": True
          },
          "date_to": {
            "type": "string",
            "description": "Data fine nel formato YYYY-MM-DD (opzionale)",
            "optional": True
          },
          "categories": {
            "type": "array",
            "description": "Lista di categorie (opzionale)",
            "optional": True
          }
        }
      },
      {
        "name": "count_articles",
        "description": "Conta quanti articoli corrispondono a determinati criteri. Usa questo per domande tipo 'quanti articoli...'",
        "parameters": {
          "author": {
            "type": "string",
            "description": "Nome autore (opzionale)",
            "optional": True
          },
          "date_from": {
            "type": "string",
            "description": "Data inizio (opzionale)",
            "optional": True
          },
          "date_to": {
            "type": "string",
            "description": "Data fine (opzionale)",
            "optional": True
          },
          "categories": {
            "type": "array",
            "description": "Categorie (opzionale)",
            "optional": True
          }
        }
      },
      {
        "name": "get_article_details",
        "description": "Recupera il contenuto completo di articoli specifici. Usa questo dopo aver trovato articoli rilevanti per leggerne il contenuto.",
        "parameters": {
          "article_ids": {
            "type": "array",
            "description": "Lista di article_id da recuperare"
          }
        }
      }
    ]
  
  def query(self, user_query: str, max_iterations: int = 10) -> Dict[str, Any]:
    """Main entry point - LLM agent handles everything"""
    
    print(f"\n{'='*80}")
    print(f"Query: {user_query}")
    print('='*80)
    
    conversation = [
      {"role": "system", "content": self._get_system_prompt()},
      {"role": "user", "content": user_query}
    ]
    
    tool_results = []
    
    for iteration in range(max_iterations):
      print(f"\n[Iteration {iteration + 1}]")
      
      # LLM decides what to do
      response = self.llm.generate_with_tools(conversation, self.tools)
      
      # If no tool calls, LLM has finished
      if not response['tool_calls']:
        print("✓ Answer generated")
        return {
          "answer": response['content'],
          "tool_calls": tool_results,
          "iterations": iteration + 1
        }
      
      # Execute tool calls
      for tool_call in response['tool_calls']:
        print(f"  → Tool: {tool_call['name']}")
        print(f"  Params: {json.dumps(tool_call['parameters'], ensure_ascii=False)}")
        
        result = self._execute_tool(
          tool_call['name'],
          tool_call['parameters']
        )
        
        tool_results.append({
          "tool": tool_call['name'],
          "parameters": tool_call['parameters'],
          "result": result
        })
        
        # Add tool result to conversation
        conversation.append({
          "role": "assistant",
          "content": response['content'] or "",
          "tool_calls": response['tool_calls']
        })
        
        conversation.append({
          "role": "tool",
          "tool_call_id": tool_call['id'],
          "name": tool_call['name'],
          "content": json.dumps(result, ensure_ascii=False, default=str)
        })
    
    return {
      "answer": f"Ho raggiunto il limite di iterazioni (${max_iterations}). Prova a riformulare la domanda.",
      "tool_calls": tool_results,
      "iterations": max_iterations
    }
  
  def _get_system_prompt(self) -> str:
    """System prompt for the agent"""
    return """Sei un assistente intelligente che aiuta a rispondere a domande su una collezione di articoli giornalistici politici italiani.

Hai accesso a vari strumenti per cercare e analizzare gli articoli. Usali in modo intelligente e strategico:

**Strategie di ricerca:**
- Per domande sul CONTENUTO ("cosa dice X su Y?", "qual è il pensiero di Z?"): usa `search_by_content` per trovare articoli rilevanti, poi `get_article_details` per leggere i contenuti completi
- Per domande su CHI ha scritto cosa: usa `search_by_author` o `search_by_metadata`
- Per CONTARE articoli: usa `count_articles`
- Puoi fare MULTIPLE chiamate agli strumenti se necessario
- Combina risultati da più strumenti per risposte complete

**Regole importanti:**
1. Basa le risposte SOLO sugli articoli trovati - MAI inventare informazioni
2. Se non trovi informazioni pertinenti negli articoli, dillo chiaramente
3. Cita SEMPRE titolo e autore degli articoli che usi come fonte
4. Se trovi molti articoli rilevanti, seleziona i più pertinenti (massimo 5-7) e leggili completamente
5. Per domande su persone menzionate negli articoli (non autori), usa search_by_content

**Formato risposta:**
- Rispondi in italiano, in modo chiaro e strutturato
- Usa le informazioni degli articoli per costruire una risposta completa
- Termina sempre citando le fonti"""
  
  def _execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
    """Execute a tool requested by the LLM"""
    
    try:
      if tool_name == "search_by_content":
        return self._tool_search_by_content(parameters)
      
      elif tool_name == "search_by_author":
        return self._tool_search_by_author(parameters)
      
      elif tool_name == "search_by_metadata":
        return self._tool_search_by_metadata(parameters)
      
      elif tool_name == "count_articles":
        return self._tool_count_articles(parameters)
      
      elif tool_name == "get_article_details":
        return self._tool_get_article_details(parameters)
      
      else:
        return {"error": f"Unknown tool: {tool_name}"}
    
    except Exception as e:
      return {"error": str(e)}
  
  def _tool_search_by_content(self, params: Dict) -> Dict:
    """Semantic search"""
    query = params['query']
    
    query_embedding = self.embedder.embed_text(query)
    
    all_articles = self.db.get_all_articles(
      projection={
        'article_id': 1,
        'metadata': 1,
        'embedding': 1,
        'url': 1,
        'source': 1
      }
    )
    
    articles_with_embeddings = [
      (art, art['embedding'])
      for art in all_articles
      if 'embedding' in art and art['embedding']
    ]
    
    if not articles_with_embeddings:
      return {"articles": [], "message": "No articles with embeddings found"}
    
    ranked = self.embedder.rank_by_similarity(
      query_embedding,
      articles_with_embeddings
    )
    
    top_articles = [
      {
        "article_id": art['article_id'],
        "title": art['metadata']['title'],
        "author": art['metadata']['author'],
        "date": art['metadata']['publication_date'].strftime('%Y-%m-%d') if art['metadata'].get('publication_date') else 'N/A',
        "url": art.get('url', 'N/A'),
        "source": art.get('source', 'N/A'),
        "similarity_score": round(float(score), 3)
      }
      for art, _, score in ranked[:self.semantic_top_k]
      if score > 0.3
    ]
    
    return {
      "total_found": len(top_articles),
      "articles": top_articles
    }
  
  def _tool_search_by_author(self, params: Dict) -> Dict:
    """Search by author"""
    author = params['author']
    
    articles = self.db.find_by_filter(
      {"metadata.author": {"$regex": author, "$options": "i"}},
      projection={'article_id': 1, 'metadata': 1, 'url': 1},
      limit=50
    )
    
    return {
      "total_found": len(articles),
      "articles": [
        {
          "article_id": a['article_id'],
          "title": a['metadata']['title'],
          "author": a['metadata']['author'],
          "date": a['metadata']['publication_date'].strftime('%Y-%m-%d') if a['metadata'].get('publication_date') else 'N/A',
          "url": a.get('url', 'N/A')
        }
        for a in articles
      ]
    }
  
  def _tool_search_by_metadata(self, params: Dict) -> Dict:
    """Search with metadata filters"""
    filters = {}
    
    if 'author' in params and params['author']:
      filters['metadata.author'] = {"$regex": params['author'], "$options": "i"}
    
    if 'date_from' in params and params['date_from']:
      if 'metadata.publication_date' not in filters:
        filters['metadata.publication_date'] = {}
      filters['metadata.publication_date']['$gte'] = datetime.fromisoformat(params['date_from'])
    
    if 'date_to' in params and params['date_to']:
      if 'metadata.publication_date' not in filters:
        filters['metadata.publication_date'] = {}
      filters['metadata.publication_date']['$lte'] = datetime.fromisoformat(params['date_to'])
    
    if 'categories' in params and params['categories']:
      filters['metadata.categories'] = {"$in": params['categories']}
    
    articles = self.db.find_by_filter(filters, limit=100)
    
    return {
      "total_found": len(articles),
      "filters_used": params,
      "articles": [
        {
          "article_id": a['article_id'],
          "title": a['metadata']['title'],
          "author": a['metadata']['author'],
          "date": a['metadata']['publication_date'].strftime('%Y-%m-%d') if a['metadata'].get('publication_date') else 'N/A'
        }
        for a in articles[:50]  # Return first 50
      ]
    }
  
  def _tool_count_articles(self, params: Dict) -> Dict:
    """Count articles"""
    filters = {}
    
    if 'author' in params and params['author']:
      filters['metadata.author'] = {"$regex": params['author'], "$options": "i"}
    
    if 'date_from' in params and params['date_from']:
      if 'metadata.publication_date' not in filters:
        filters['metadata.publication_date'] = {}
      filters['metadata.publication_date']['$gte'] = datetime.fromisoformat(params['date_from'])
    
    if 'date_to' in params and params['date_to']:
      if 'metadata.publication_date' not in filters:
        filters['metadata.publication_date'] = {}
      filters['metadata.publication_date']['$lte'] = datetime.fromisoformat(params['date_to'])
    
    if 'categories' in params and params['categories']:
      filters['metadata.categories'] = {"$in": params['categories']}
    
    count = self.db.count_by_filter(filters)
    
    return {
      "count": count,
      "filters_used": params
    }
  
  def _tool_get_article_details(self, params: Dict) -> Dict:
    """Get full article content"""
    article_ids = params['article_ids']
    
    articles = self.db.find_by_filter(
      {"article_id": {"$in": article_ids}},
      projection={
        'article_id': 1,
        'metadata': 1,
        'content.full_text': 1,
        'url': 1,
        'source': 1
      }
    )
    
    return {
      "total_found": len(articles),
      "articles": [
        {
          "article_id": a['article_id'],
          "title": a['metadata']['title'],
          "author": a['metadata']['author'],
          "date": a['metadata']['publication_date'].strftime('%Y-%m-%d') if a['metadata'].get('publication_date') else 'N/A',
          "url": a.get('url', 'N/A'),
          "source": a.get('source', 'N/A'),
          "content": a['content']['full_text']
        }
        for a in articles
      ]
    }