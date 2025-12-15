#!/usr/bin/env python3
"""
Query

Run queries against the RAG system
"""

from rag_journal.rag.query_router import QueryRouter


def print_result(result):
  """Pretty print query result"""
  
  print(f"Tipo del risultato: {result.get('type', 'unknown')}")
  
  # Print answer
  print(f"Risposta:\n")
  print(result.get('answer', 'Nessuna risposta generata'))
  
  # Handle different result types
  result_type = result.get('type')
  articles = []

  if result_type == 'count':
    print(f"\nRisultati: {result.get('result', 0)}")
  
  elif result_type in ['no_results', 'low_relevance']:
    # Don't print article list for these cases
    print(f"\n⚠ Nessun risultato rilevante trovato")
    if 'best_score' in result:
      print(f"  Miglior punteggio: {result['best_score']:.3f}")
  
  elif result_type in ['semantic_results', 'hybrid_results']:
    articles = result.get('result', [])
  
  if not articles:
    print(f"\n⚠ Nessun articolo trovato")
    return
  
  print(f"\nTrovati {len(articles)} articoli rilevanti:")
  
  # Show best score
  if 'best_score' in result:
    print(f"  Miglior punteggio di rilevanza: {result['best_score']:.3f}")
  
    #for i, item in enumerate(articles[:3], 1):  # Show top 3
    for i, item in enumerate(articles, 1):
      art = item['article']
      score = item['score']
      print(f"\n[{i}] {art['metadata']['title']}")
      print(f"  Autore: {art['metadata']['author']}")
      
      # Safely format date
      pub_date = art['metadata'].get('publication_date')
      date_str = pub_date.strftime('%Y-%m-%d') if pub_date else 'N/A'
      print(f"  Date: {date_str}")
      
      print(f"  Similarity: {score:.3f}")
      
      # Show URL if available
      url = art.get('url')
      if url:
         print(f"    URL: {url}")
      
      # Show source if available
      source = art.get('source')
      if source:
        print(f"    Fonte: {source}")

      # Show number if available
      number = art.get('number')
      if number:
        print(f"    Numero: {number}")
      
      # Show preview
      text = art['content'].get('summary', art['content']['full_text'][:200])
      if text:
        print(f"  Preview: {text[:150]}...")

  elif result_type == 'list':
    articles = result.get('result', [])
    print(f"\nElenco degli articoli:")
    
    for i, art in enumerate(articles[:10], 1): # Show first 10
      url = art.get('url', '')
      url_display = f" - {url}" if url else ""
      print(f"{i}. {art['metadata']['title']} - {art['metadata']['author']} - {art.get('url')} - {art.get('source')} - {art.get('number')}")


def main():
  """Test the RAG query system"""
  
  # Initialize router
  router = QueryRouter()
  
  while True:
    try:
      query = input(f"\nFai una domanda ('esci' per terminare): ").strip()
    except KeyboardInterrupt:
      print("\nInterruzione da tastiera, si termina")
      exit(0)
    except EOFError:
      print("\nFine dell'input, si termina")
      exit(0)
    
    if query.lower() in ['esci', 'quit', 'exit', 'q']:
      print("Arrivederci!")
      exit(0)
    
    if not query:
      continue
    
    # Execute query
    try:
      result = router.query(query)
      print_result(result)
    except Exception as e:
      print(f"✗ Errore: {e}")


if __name__ == "__main__":
  main()
