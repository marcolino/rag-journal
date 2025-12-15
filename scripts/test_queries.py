#!/usr/bin/env python3
"""
Test Queries Script

Run test queries against the RAG system
"""

import click
from colorama import Fore, Style, init
from rag_journal.rag.query_router import QueryRouter


# Initialize colorama
init(autoreset = True)


def print_result(result):
  """Pretty print query result"""
  
  print(f"\n{Fore.CYAN}{'='*80}")
  print(f"RESULT TYPE: {result.get('type', 'unknown')}")
  print('='*80 + Style.RESET_ALL)
  
  # Print answer
  print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
  print(result.get('answer', 'No answer generated'))
  
  # Print additional info based on type
  if result['type'] == 'count':
    print(f"\n{Fore.YELLOW}Count:{Style.RESET_ALL} {result.get('result', 0)}")
  
  elif result['type'] in ['semantic_results', 'hybrid_results']:
    articles = result.get('result', [])
    print(f"\n{Fore.YELLOW}Found {len(articles)} relevant articles:{Style.RESET_ALL}")
    
    #for i, item in enumerate(articles[:3], 1):  # Show top 3
    for i, item in enumerate(articles, 1):
      art = item['article']
      score = item['score']
      print(f"\n{Fore.MAGENTA}[{i}] {art['metadata']['title']}{Style.RESET_ALL}")
      print(f"  Author: {art['metadata']['author']}")
      
      # Safely format date
      pub_date = art['metadata'].get('publication_date')
      date_str = pub_date.strftime('%Y-%m-%d') if pub_date else 'N/A'
      print(f"    Date: {date_str}")
      
      print(f"    Similarity: {score:.3f}")

      # Show URL if available
      url = art.get('url')
      if url:
        print(f"    URL: {Fore.BLUE}{url}{Style.RESET_ALL}")

      # Show source if available
      source = art.get('source')
      if source:
        print(f"    Source: {Fore.BLUE}{source}{Style.RESET_ALL}")

      # Show number if available
      number = art.get('number')
      if number:
        print(f"    Number: {Fore.BLUE}{number}{Style.RESET_ALL}")

      # Show preview
      text = art['content'].get('summary', art['content']['full_text'][:200])
      if (text): print(f"  Preview: {text[:150]}...")
  
  elif result['type'] == 'list':
    articles = result.get('result', [])
    print(f"\n{Fore.YELLOW}Listing articles:{Style.RESET_ALL}")
    
    for i, art in enumerate(articles[:10], 1):  # Show first 10
      url = art.get('url', 'N/A')
      source = art.get('source', 'N/A')
      number = art.get('number', 'N/A')
      print(f"{i}. {art['metadata']['title']} - {art['metadata']['author']} - {url} - {source} - {number}")


@click.command()
@click.option('--interactive/--batch', default = True, help = 'Interactive mode or batch test queries')
def main(config = 'config/config.yaml', interactive = True):
  """Test the RAG query system"""
  
  print('='*80)
  print("RAG QUERY TESTING SYSTEM")
  print('='*80 + Style.RESET_ALL)
  
  # Initialize router
  router = QueryRouter()
  
  if interactive:
    # Interactive mode
    print(f"\n{Fore.GREEN}Interactive mode - Type 'quit' to exit{Style.RESET_ALL}")
    
    while True:
      print(f"\n{Fore.YELLOW}{'─'*80}{Style.RESET_ALL}")
      query = input(f"{Fore.CYAN}Enter your query: {Style.RESET_ALL}").strip()
      
      if query.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
      
      if not query:
        continue
      
      # Execute query
      try:
        result = router.query(query)
        print_result(result)
      except Exception as e:
        print(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}")
  
  else:
    # Batch mode - test predefined queries
    test_queries = [
      "Quanti articoli ha scritto Mario Rossi nel 2023?",
      "Cosa dice Giulia Verdi sulla guerra in Ucraina?",
      "Quali articoli parlano di energia rinnovabile?",
      "Qual è il trend della politica energetica europea dal 2022?",
      "Articoli su decarbonizzazione e cambiamento climatico",
      "Chi ha scritto più articoli sulla NATO?",
    ]
    
    print(f"\n{Fore.GREEN}Running {len(test_queries)} test queries...{Style.RESET_ALL}\n")
    
    for i, query in enumerate(test_queries, 1):
      print(f"\n{Fore.MAGENTA}{'='*80}")
      print(f"TEST QUERY {i}/{len(test_queries)}")
      print('='*80 + Style.RESET_ALL)
      print(f"{Fore.CYAN}Query: {query}{Style.RESET_ALL}")
      
      try:
        result = router.query(query)
        print_result(result)
      except Exception as e:
        print(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}")
      
      # Separator
      print(f"\n{Fore.YELLOW}{'─'*80}{Style.RESET_ALL}")


if __name__ == "__main__":
  main()
