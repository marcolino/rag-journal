import pymongo
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from datetime import datetime

class QueryRouter:
  def __init__(self):
    # Small LLM for query classification (runs on CPU)
    self.classifier_model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B-Instruct",  # or "microsoft/Phi-3-mini-4k-instruct"
      torch_dtype=torch.float32,
      device_map="cpu"
    )
    self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    # Embedding model for semantic search (multilingual, 384 dims)
    self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # MongoDB connection
    self.db = pymongo.MongoClient("mongodb://localhost:27017/")["journal"]
    
  def classify_query(self, user_query):
    """Classify query type and extract parameters"""
    
    prompt = f"""Analyze questa domanda in italiano e determina il tipo di query necessaria.

Domanda: {user_query}

Rispondi SOLO con un JSON valido in questo formato:
{{
  "query_type": "metadata" | "semantic" | "hybrid" | "analytical",
  "requires_count": true/false,
  "filters": {{
  "author": "nome autore o null",
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} o null,
  "categories": ["categoria1"] o null
  }},
  "semantic_query": "query per ricerca semantica o null",
  "reasoning": "breve spiegazione"
}}

Esempi:
- "Quanti articoli di Mario Rossi nel 2023?" → metadata + count
- "Cosa dice Giulia Verdi sull'Ucraina?" → hybrid (filter author + semantic search)
- "Quali articoli parlano di decarbonizzazione?" → semantic
- "Trend della guerra in Ucraina dal 2022" → analytical (semantic + temporal ordering)
"""

    messages = [{"role": "user", "content": prompt}]
    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = self.tokenizer(text, return_tensors="pt")
    outputs = self.classifier_model.generate(
      **inputs,
      max_new_tokens=512,
      temperature=0.1,
      do_sample=True
    )
    
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
      json_start = response.find('{')
      json_end = response.rfind('}') + 1
      json_str = response[json_start:json_end]
      return json.loads(json_str)
    except:
      # Fallback to semantic search
      return {
        "query_type": "semantic",
        "requires_count": False,
        "filters": {},
        "semantic_query": user_query,
        "reasoning": "Could not parse query, defaulting to semantic search"
      }
  
  def execute_query(self, user_query):
    """Main entry point: classify and execute"""
    
    classification = self.classify_query(user_query)
    print(f"Query classified as: {classification['query_type']}")
    print(f"Reasoning: {classification['reasoning']}")
    
    query_type = classification['query_type']
    
    if query_type == "metadata":
      return self.execute_metadata_query(classification)
    elif query_type == "semantic":
      return self.execute_semantic_query(classification)
    elif query_type == "hybrid":
      return self.execute_hybrid_query(classification)
    elif query_type == "analytical":
      return self.execute_analytical_query(classification)
  
  def build_mongo_filter(self, filters):
    """Build MongoDB filter from extracted parameters"""
    mongo_filter = {}
    
    if filters.get('author'):
      mongo_filter['metadata.author'] = filters['author']
    
    if filters.get('date_range'):
      date_filter = {}
      if filters['date_range'].get('start'):
        date_filter['$gte'] = datetime.fromisoformat(filters['date_range']['start'])
      if filters['date_range'].get('end'):
        date_filter['$lte'] = datetime.fromisoformat(filters['date_range']['end'])
      if date_filter:
        mongo_filter['metadata.publication_date'] = date_filter
    
    if filters.get('categories'):
      mongo_filter['metadata.categories'] = {'$in': filters['categories']}
    
    return mongo_filter
  
  def execute_metadata_query(self, classification):
    """Handle pure metadata queries (counting, filtering)"""
    
    mongo_filter = self.build_mongo_filter(classification['filters'])
    
    if classification['requires_count']:
      count = self.db.articles.count_documents(mongo_filter)
      return {
        "type": "count",
        "result": count,
        "answer": f"Ho trovato {count} articoli che corrispondono ai criteri."
      }
    else:
      articles = list(self.db.articles.find(
        mongo_filter,
        {'metadata': 1, 'article_id': 1}
      ).limit(20))
      
      return {
        "type": "list",
        "result": articles,
        "answer": f"Ho trovato {len(articles)} articoli (mostrando i primi 20)."
      }
  
  def execute_semantic_query(self, classification):
    """Handle semantic search queries"""
    
    query_text = classification['semantic_query']
    query_embedding = self.embedder.encode(query_text).tolist()
    
    # MongoDB vector search (requires MongoDB Atlas or local vector search setup)
    # For local MongoDB, we'll use a simple similarity calculation
    
    # Fetch all articles (for 20k articles, this is feasible with 16GB RAM)
    # In production, use MongoDB Atlas Vector Search or add pagination
    articles = list(self.db.articles.find({}, 
      {'article_id': 1, 'metadata': 1, 'content.summary': 1, 'embedding': 1}
    ))
    
    # Calculate cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
      return dot(a, b) / (norm(a) * norm(b))
    
    scored_articles = []
    for article in articles:
      if 'embedding' in article:
        similarity = cosine_similarity(query_embedding, article['embedding'])
        scored_articles.append({
          'article': article,
          'score': similarity
        })
    
    # Sort by score
    scored_articles.sort(key=lambda x: x['score'], reverse=True)
    top_articles = scored_articles[:5]
    
    return {
      "type": "semantic_results",
      "result": top_articles,
      "answer": f"Ho trovato {len(top_articles)} articoli rilevanti."
    }
  
  def execute_hybrid_query(self, classification):
    """Combine metadata filtering + semantic search"""
    
    # First filter by metadata
    mongo_filter = self.build_mongo_filter(classification['filters'])
    
    # Get filtered articles
    articles = list(self.db.articles.find(
      mongo_filter,
      {'article_id': 1, 'metadata': 1, 'content.summary': 1, 'content.full_text': 1, 'embedding': 1}
    ))
    
    if not articles:
      return {
        "type": "no_results",
        "result": [],
        "answer": "Nessun articolo trovato con questi filtri."
      }
    
    # Then do semantic search on filtered set
    query_text = classification['semantic_query']
    query_embedding = self.embedder.encode(query_text).tolist()
    
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
      return dot(a, b) / (norm(a) * norm(b))
    
    scored_articles = []
    for article in articles:
      if 'embedding' in article:
        similarity = cosine_similarity(query_embedding, article['embedding'])
        scored_articles.append({
          'article': article,
          'score': similarity
        })
    
    scored_articles.sort(key=lambda x: x['score'], reverse=True)
    top_articles = scored_articles[:5]
    
    # Generate answer using LLM
    answer = self.generate_answer(user_query=classification['semantic_query'], 
                     articles=top_articles)
    
    return {
      "type": "hybrid_results",
      "result": top_articles,
      "answer": answer
    }
  
  def execute_analytical_query(self, classification):
    """Handle complex analytical queries (trends, comparisons)"""
    
    # Similar to hybrid but with temporal ordering and aggregation
    return self.execute_hybrid_query(classification)
  
  def generate_answer(self, user_query, articles):
    """Generate final answer from retrieved articles"""
    
    # Prepare context from articles
    context = "\n\n".join([
      f"Articolo: {a['article']['metadata']['title']}\n"
      f"Autore: {a['article']['metadata']['author']}\n"
      f"Data: {a['article']['metadata']['publication_date']}\n"
      f"Contenuto: {a['article']['content'].get('summary', a['article']['content'].get('full_text', '')[:1000])}\n"
      f"Rilevanza: {a['score']:.2f}"
      for a in articles[:3]  # Top 3 articles
    ])
    
    prompt = f"""Basandoti SOLO sugli articoli seguenti, rispondi alla domanda dell'utente.

Articoli:
{context}

Domanda: {user_query}

Rispondi in modo conciso e cita sempre gli articoli fonte (titolo e autore).
"""

    messages = [{"role": "user", "content": prompt}]
    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = self.tokenizer(text, return_tensors="pt")
    outputs = self.classifier_model.generate(
      **inputs,
      max_new_tokens=512,
      temperature=0.7,
      do_sample=True
    )
    
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (after the prompt)
    answer_start = response.find("Rispondi in modo conciso")
    if answer_start != -1:
      answer = response[answer_start:].split("\n", 1)[-1].strip()
    else:
      answer = response.split("assistant")[-1].strip()
    
    return answer


# Usage
router = QueryRouter()

# Test queries
queries = [
  "Quanti articoli ha scritto Mario Rossi nel 2023?",
  "Cosa dice Giulia Verdi sulla guerra in Ucraina?",
  "Quali articoli parlano di decarbonizzazione?",
  "Qual è il trend della politica energetica europea dal 2022?"
]

for query in queries:
  print(f"\n{'='*80}")
  print(f"Query: {query}")
  print('='*80)
  result = router.execute_query(query)
  print(f"\nAnswer: {result['answer']}")