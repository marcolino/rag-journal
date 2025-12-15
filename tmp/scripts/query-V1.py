#!/usr/bin/env python3
"""
Query

Run queries against the RAG system
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import click
#from colorama import Fore, Style, init
from rag_journal.rag.agentic_rag import AgenticRAG

load_dotenv('.env')
#init(autoreset=True)

@click.command()
@click.option('--config', default='config/config.yaml', help='Config file path')
@click.option('--interactive/--single', default=True, help='Interactive mode or single query')
@click.argument('query', required=False)
def main(config, interactive, query):
  """Query the Agentic RAG system"""
  
  print('='*80)
  print("AGENTIC RAG QUERY SYSTEM")
  print('='*80)
  
  # Initialize system
  rag = AgenticRAG(config)
  
  if interactive:
    print(f"\nModalità interattiva ('exit' per terminare)")
    
    while True:
      print(f"\n{'─'*80}")
      user_query = input(f"La tua domanda: ").strip()
      
      if user_query.lower() in ['exit', 'quit', 'q']:
        print("Arrivederci.")
        break
      
      if not user_query:
        continue
      
      try:
        result = rag.query(user_query)
        
        print(f"\n{'='*80}")
        print("RISPOSTA")
        print('='*80)
        print(result['answer'])
        
        print(f"\nStats:")
        print(f"  Iterazioni: {result['iterations']}")
        print(f"  Tools usati: {len(result['tool_calls'])}")
        
      except Exception as e:
        print(f"✗ Errore: {e}")
        import traceback
        traceback.print_exc()
  
  else:
    if not query:
      print(f"Errore: Query necessaria im modalità non-interattiva")
      sys.exit(1)
    
    try:
      result = rag.query(query)
      
      print(f"\n{'='*80}")
      print("RISPOSTA")
      print('='*80)
      print(result['answer'])
      
      print(f"\nStats:")
      print(f"  Iterazioni: {result['iterations']}")
      print(f"  Tools usati: {len(result['tool_calls'])}")
      
    except Exception as e:
      print(f"✗ Errore: {e}")
      sys.exit(1)


if __name__ == "__main__":
  main()
  