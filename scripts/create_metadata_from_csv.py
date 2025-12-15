#!/usr/bin/env python3
"""
Create JSON metadata files from CSV

CSV format:
article_id,author,publication_date,title,categories,translator
article_001,Mario Rossi,2023-05-15,Il titolo,"politica,ue",
article_002,Giulia Verdi,2023-06-20,Altro titolo,"ambiente,energia",Giovanni Bianchi

Categories should be semicolon-separated: "cat1;cat2;cat3"
"""

import csv
import json
import click
from pathlib import Path


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--output-dir', default='data/metadata', help='Output directory for JSON files')
@click.option('--delimiter', default=',', help='CSV delimiter')
def main(csv_file, output_dir, delimiter):
  """Create JSON metadata files from CSV"""
  
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  
  print(f"Reading CSV: {csv_file}")
  print(f"Output directory: {output_dir}")
  print("="*80)
  
  created = 0
  errors = 0
  
  with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=delimiter)
    
    # Verify required columns
    required = {'article_id', 'author', 'publication_date', 'title', 'categories'}
    if not required.issubset(set(reader.fieldnames)):
      print(f"✗ Error: CSV must contain columns: {required}")
      print(f"  Found: {reader.fieldnames}")
      return
    
    for row in reader:
      try:
        article_id = row['article_id']
        
        # Parse categories
        categories = [c.strip() for c in row['categories'].split(';') if c.strip()]
        
        # Build metadata
        metadata = {
          "author": row['author'],
          "publication_date": row['publication_date'],  # Must be YYYY-MM-DD
          "categories": categories,
          "translator": row.get('translator') if row.get('translator') else None,
          "title": row['title']
        }
        
        # Write JSON file
        output_file = output_path / f"{article_id}.json"
        with open(output_file, 'w', encoding='utf-8') as out:
          json.dump(metadata, out, indent=2, ensure_ascii=False)
        
        created += 1
        
        if created % 100 == 0:
          print(f"  Created {created} files...")
          
      except Exception as e:
        print(f"✗ Error processing {row.get('article_id', 'unknown')}: {e}")
        errors += 1
  
  print("="*80)
  print(f"✓ Created: {created} metadata files")
  if errors > 0:
    print(f"✗ Errors: {errors}")
  print(f"\nMetadata files saved to: {output_dir}/")


@click.command()
@click.option('--output-file', default='metadata_template.csv', help='Output CSV template file')
def create_template(output_file):
  """Create a CSV template for metadata"""
  
  template = [
    {
      'article_id': 'example_001',
      'author': 'Mario Rossi',
      'publication_date': '2023-05-15',
      'title': 'Esempio di articolo',
      'categories': 'politica;economia;europa',
      'translator': ''
    },
    {
      'article_id': 'example_002',
      'author': 'Giulia Verdi',
      'publication_date': '2023-06-20',
      'title': 'Altro esempio',
      'categories': 'ambiente;energia',
      'translator': 'Giovanni Bianchi'
    }
  ]
  
  with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=template[0].keys())
    writer.writeheader()
    writer.writerows(template)
  
  print(f"✓ Template created: {output_file}")
  print("\nFormat:")
  print("  - article_id: unique ID matching .txt filename")
  print("  - author: full name")
  print("  - publication_date: YYYY-MM-DD format")
  print("  - title: article title")
  print("  - categories: semicolon-separated (cat1;cat2;cat3)")
  print("  - translator: name or leave empty")


if __name__ == "__main__":
  import sys
  
  if len(sys.argv) > 1 and sys.argv[1] == 'template':
    create_template()
  else:
    main()
