#!/usr/bin/env python3
"""
Database Setup Script

Initialize MongoDB database and verify connection
"""

import click
from rag_journal.database.mongodb_client import MongoDBClient


@click.command()
@click.option('--config', default='config/config.yaml', help='Config file path')
def main(config):
  """Setup and verify database"""
  
  print("="*80)
  print("DATABASE SETUP")
  print("="*80)
  
  try:
    # Initialize client
    print("\nConnecting to MongoDB...")
    db = MongoDBClient()
    
    # Get statistics
    stats = db.get_statistics()
    
    print("\n✓ Database connection successful!")
    print("\nCurrent Statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Unique authors: {stats['unique_authors']}")
    
    if stats['date_range']['oldest']:
      print(f"  Date range: {stats['date_range']['oldest'].date()} to {stats['date_range']['newest'].date()}")
    else:
      print("  Date range: No articles yet")
    
    print("\n✓ Database is ready for use!")
    
  except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Ensure MongoDB is running (mongod)")
    print("  2. Check connection string in config/config.yaml")
    print("  3. Verify network connectivity")


if __name__ == "__main__":
  main()
  