#!/usr/bin/env python3
"""
Query Script for Agentic RAG System
"""

import click
from dotenv import load_dotenv
from rag_journal.utils.logger import setup_logger, logger
from rag_journal.rag.agentic_rag import AgenticRAG


load_dotenv('.env')
setup_logger(
  log_file="logs/rag.log",
  level="INFO" # or DEBUG for more details
)

@click.command()
@click.option('--mode', type=click.Choice(['single', 'chat']), default='chat', 
              help='Query mode: single question or chat conversation')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.argument('query', required=False)

def main(mode, debug, query):
  """Query the Agentic RAG system"""
  
  if debug:
    logger.setLevel("DEBUG")
    logger.debug("Debug mode enabled")
    
  logger.info("Starting RAG query system")

  # Initialize system
  rag = AgenticRAG()
  
  if query:
    # Single query from command line
    logger.info("Query singola da riga comando - Modalità batch")
    result = rag.query(query)
    print(f"\nAnswer: {result['answer']}")
    logger.info(f"Richiesta completata ({result['iterations']} iterazioni)")
    logger.info(f"Strumenti utilizzati: {len(result['tool_calls'])}")
    return
  
  # Interactive mode
  if mode == 'chat':
    print("\nModalità chat - History di conversazine mantenuta")
    print("Comandi: 'exit' per terminare, 'reset' per pulire l'history\n")
  else:
    print("\nModalità query singola - History di conversazione non mantenuta")
    print("Premi 'exit' per terminare\n")
  
  while True:
    print("-"*80)
    user_query = input("Tu: ").strip()
    
    if user_query.lower() in ['exit', 'quit', 'q']:
      print("Arrivederci!")
      break
    
    if user_query.lower() == 'reset' and mode == 'chat':
      rag.reset_chat()
      continue
    
    if user_query.lower() == 'history' and mode == 'chat':
      history = rag.get_chat_history()
      print(f"\nChat history ({len(history)} messaggi):")
      for i, msg in enumerate(history, 1):
        print(f"  {i}. {msg['role']}: {msg['content'][:100]}...")
      continue
    
    if not user_query:
      continue
    
    try:
      if mode == 'chat':
        result = rag.chat(user_query)
      else:
        result = rag.query(user_query)
      
      print(f"\nAssistente: {result['answer']}\n")
      
      if mode == 'single':
        logger.info(f"[Statistiche: {result['iterations']} iterazioni, {len(result['tool_calls'])} tools]")
    
    except Exception as e:
      logger.error(f"\n✗ Errore: {e}\n")
      import traceback
      traceback.print_exc()


if __name__ == "__main__":
  main()
