#!/usr/bin/env python

"""
Articles download script
"""

import os
import re
import sys
import glob
import html
import html5lib
import yaml
import numpy as np
import pickle
import faiss
import requests
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin #, urlparse
from bs4 import BeautifulSoup
from requests_file import FileAdapter # DEV ONLY
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.model_article import Article
from src.utils import data_dump, data_load
from src.config import Config

class ArticleManager:
  """Manages article loading, storage and state"""
  def __init__(self, env: Optional[str] = None, log_level: int = logging.INFO):
    self.logger = logging.getLogger(f"{__name__}.ArticleManager")
    self.log_level = log_level
    
    # Instance state instead of global state
    self._status: Dict = {}
    self._articles: List = []

  @property
  def status(self) -> Dict:
    """Get current status dictionary"""
    return self._status.copy() # Return copy to prevent external modification
    
  @property
  def articles(self) -> List:
    """Get current articles list"""
    return self._articles.copy() # Return copy to prevent external modification

  # Force yaml dump a literal multiline string representation in "|" style
  def _setup_yaml_representer(self):
    def literal_str_representer(dumper, data):
      if '\n' in data: # check if multiline
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style = "|")
      return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    yaml.add_representer(str, literal_str_representer)

  def articles_download(self) -> None:
    """Download all articles in a number."""
    self._status = data_load()
    try:
      n = self._status["last_article_number"]
    except KeyError:
      n = 0

    m = 0
    while True:
      n += 1
      target_url = Config.ARTICLES_SOURCE_URL_PATTERN % n
      html, base_url = self.articles_download_index(target_url) # Download the index page for the current article number
      if html:
        links = self.parse_html(html, base_url)
        count = len(links)
        if count > 0:
          m += 1
          self.links_download(links, n)
        else:
          n = n - 1 # we ignore the last article, "in preparazione" ...
          break
      else:
        break

    self._status["last_article_number"] = n
    data_dump(self._status)

    self.logger.info(f"Downloaded {m} new articles from {Config.ARTICLES_SOURCE_URL_BASE}")
    return m

  def articles_download_index(self, url) -> [str, str]:
    """Download the HTML content of all articles in a number."""
    try:
      os.makedirs(Config.ARTICLES_LOCAL_FOLDER, exist_ok = True)
    except OSError as e:
      self.logger.error(e, exc_info = self.log_level <= logging.DEBUG)
      return None, url

    try:
      response = requests.get(url, timeout = Config.REQUEST_TIMEOUT_SECONDS)
      response.raise_for_status()
      html_content = response.text
      return html_content, url
    except requests.RequestException as e:
      if response.status_code != 404: # 404 is expected for the last page
        self.logger.error(e, exc_info = self.log_level <= logging.DEBUG)
      return None, url

  def parse_html(self, html, base_url) -> List:
    """Parse the HTML content and extract links to articles."""
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for tag in soup.find_all('a', href = True):
      href = tag['href']
      full_url = urljoin(base_url, href)
      if full_url.startswith('http'):
        # Only download links from the same domain
        if full_url.startswith(Config.ARTICLES_SOURCE_URL_BASE):
          # Only download links that do match the expected frame pattern
          if not re.compile(Config.ARTICLES_SOURCE_NEWS_PATTERN).search(full_url):
            links.add(full_url)
    return links

  def links_download(self, links, n) -> None:
    """Download articles which are "new" from the links extracted from the article index."""
    for link in links:
      if any(article.get('url') == link for article in self._articles): # Skip articles already downloaded
        self.logger.info(f"Skipped existing {link}")
      else:
        try:
          # Get path and filename from the link
          categories_and_filename = link.replace(f"{Config.ARTICLES_SOURCE_URL_BASE}/{Config.ARTICLES_SOURCE_URL_PATH}/", '')
          categories = categories_and_filename.split('/') # Get the categories from the link
          filename = categories.pop() # Build the filename from the categories

          filename = f"{"_".join(categories)}_{filename}"
          basename, ext = os.path.splitext(filename)
          filename = basename + Config.ARTICLES_LOCAL_EXTENSION # Change the extension
          filepath = os.path.join(Config.ARTICLES_LOCAL_FOLDER, filename)

          # Download and dump article

          # Check if article exists #, or if we have to force overwrite
          if os.path.exists(filepath):
            raise ValueError(f"Article at {filepath} exists already, but it's link {link} was not found in articles cache")

          try:
            # Download article
            # DEV ONLY: READ FILE FROM FILE SYSTEM OR FROM THE INTERNET ###########################################
            url_or_filename = None
            if (Config.ARTICLES_SOURCE_FILE):
              session = requests.Session()
              session.mount('file://', FileAdapter())
              url_or_filename = f"file://{os.getcwd()}/articles_html/{os.path.basename(link)}"
              response = session.get(url_or_filename, timeout = Config.REQUEST_TIMEOUT_SECONDS)
            else:
              url_or_filename = link
              response = requests.get(url_or_filename, timeout = Config.REQUEST_TIMEOUT_SECONDS)
            #######################################################################################################
            response.raise_for_status() # Check if the response is valid, raise exception if not
          except requests.RequestException as e:
            self.logger.error(e, exc_info = self.log_level <= logging.DEBUG)
            continue

          article = self.convert_to_text(response, n)
          try: # Dump article
            with open(filepath, 'w', encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
              yaml.dump(article, f, allow_unicode = True, sort_keys = False)
          except Exception as e:
            self.logger.error(e, exc_info = self.log_level <= logging.DEBUG)
            continue

          self.logger.info(f"Downloaded {link}")

        except Exception as e:
          self.logger.error(e, exc_info = self.log_level <= logging.DEBUG)
          raise

  def convert_to_text(self, response, n) -> Dict: # TODO: -> Article
    # Browsers treat charset=iso-8859-1 encoding as Windows-1252.
    # This behavior is a de facto standard in browsers to maintain compatibility with legacy content.
    if response.encoding == 'ISO-8859-1':
      response.encoding = 'windows-1252'

    html = response.text

    # Preprocess html to solve quirks in the source
    html = html.replace("’", "'") # replace ’ with ' (in this site the typographic close quote is wrongly used for the regular 
    html = html.replace("–", "'") # replace – with -
    html = html.replace(r'\302\240', r'') # Remove non-breaking spaces

    # Try to match header fields
    article = Article()
    #title = ""
    #category = ""
    ##topics = ""
    ##date_publishing_resistenze = ""
    ##date_publishing_source = ""
    #number = ""
    ##author = ""
    ##source = ""
    ##translator = ""
    #contents = ""
    
    # Match title
    match_title = re.search(f'<title>(.*?)</title>', html, flags = re.IGNORECASE | re.DOTALL) # match title
    if match_title:
      article.title = match_title.group(1)

    article.categories = ["po", "ar"] # TODO...

    # Match category, number and date_publishing_resistenze
    category = None
    match_header = re.search(
      f'<a href="https?://{Config.ARTICLES_SOURCE_URL_DOMAIN}/">{Config.ARTICLES_SOURCE_URL_DOMAIN}</a>[\\s\\-]*(.*?)<br\\s*/?>',
      html,
      flags = re.IGNORECASE | re.DOTALL
    )
    if match_header:
      header = match_header.group(1)
      html = html.replace(match_header.group(0), "") # Remove header from html
      parts_separator = " - "
      header = header.strip().replace("<i>", "").replace("</i>", "")
      parts = [p.strip() for p in header.split(parts_separator)]
        
      if len(parts) < 3: 
        # Input string must have at least 3 groups separated by '{parts_separator}'
        # Ignore error and keep the original category, number, date
        pass
      else:
        category = parts[0].strip()
        number = parts[-1].strip()
        number = re.sub(r"n\.\s?", "", number) # Remove "n." from number
        date = parts[-2].strip()
        from datetime import datetime
        try:
          date = datetime.strptime(date, "%d-%m-%y").strftime("%Y-%m-%d")
        except ValueError:
          self.logger.info(f"Unforeseen date format: {date}")
          # Ignore error and keep the original date
          pass
        date_publishing_resistenze = date

      topics = [s.strip() for s in parts[1:-2]] # All groups between first and last two
      
    # Remove unwanted parts of the html
    html = html.replace(r"&nbsp;", " ") # Remove non-breaking spaces

    # Remove repeated title in body
    html = re.sub(r"<div><font size=\"4\">.*?<\/font><\/div>", "", html, flags = re.IGNORECASE | re.DOTALL)

    # Match author
    author = None
    author_pattern = re.compile(
      r"""<div><font\s+size=["']3["'].*?>\s*<br\s*/?>\s*     # anchor start
          ([^|<]+?)\s*\|\s*                               # capture author (text before pipe)
          (.+?)\s*                                        # capture source (including tags)
          <br\s*/?>                                       # ending <br />
      """,
      re.IGNORECASE | re.VERBOSE | re.DOTALL
    )
    match_author_and_source = author_pattern.search(html)
    if match_author_and_source:
      author = match_author_and_source.group(1)
      source = match_author_and_source.group(2) if match_author_and_source.group(2) else ""
      # print("Author:", author)
      # print("Source:", source)
      html = html.replace(match_author_and_source.group(0), "") # Remove author from html

    # Match date_publishing_source in format dd/mm/yyyy
    date_publishing_source = None
    match_date = re.search(
      r"\s*(\d\d)/(\d\d)/(\d\d\d\d)\s*<br\s*/?>",
      html,
      flags = re.IGNORECASE | re.DOTALL
    )
    if match_date:
      date1 = match_date.group(1)
      date2 = match_date.group(2)
      date3 = match_date.group(3)
      date_publishing_source = f"{date3}-{date2}-{date1}"
      html = html.replace(match_date.group(0), "") # Remove date from html

    # Match date_publishing_source in format yyyy
    match_date = re.search(
      r"\s*(\d\d\d\d)\s*<br\s*/?>",
      html,
      flags = re.IGNORECASE | re.DOTALL
    )
    if match_date:
      date1 = match_date.group(1)
      date_publishing_source = f"{date1}"
      html = html.replace(match_date.group(0), "") # Remove date from html

    # Match date_publishing_source in format "italian month name"
    match_date = re.search(
      r"\s*(Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre)\s*<br\s*/?>",
      html,
      flags = re.IGNORECASE | re.DOTALL
    )
    if match_date:
      date1 = match_date.group(1)
      date_publishing_source = f"{date1}"
      html = html.replace(match_date.group(0), "") # Remove date from html

    # Match translator
    translator = None
    match_translator = re.search(
      r"(?:Traduzione|Trascrizione)\s*(?:di\s+(?:.*?)?)?\s*(?:per\s*<a .*?>.*?</a>)?\s*(?:a\s*cura\s*(?:di|del|della|dello|degli|dei)\s*(.*?))<br\s*/?>",
      html,
      flags = re.IGNORECASE | re.DOTALL
    )
    if match_translator:
      translator = match_translator.group(1)
      html = html.replace(match_translator.group(0), "") # Remove translator from html

    # Remove footer parts
    html = re.sub(r"<table.*(sostieni resistenze|fai una donazione).*</table>", "", html, flags = re.IGNORECASE | re.DOTALL)

    # Parse html to extract only text
    article.url = response.url
    article.title = self.html_to_text(article.title)
    #category = self.html_to_text(category)
    #topics = topics
    #date_publishing_resistenze = self.html_to_text(date_publishing_resistenze)
    number = self.html_to_text(number)
    #author = self.html_to_text(author)
    #source = self.html_to_text(source)
    #date_publishing_source = self.html_to_text(date_publishing_source)
    #translator = self.html_to_text(translator)
    text = self.html_to_text(html, tag_name = "body")

    if not int(number) == int(n):
      #self.logger.warning(f"Real number is {n}, extracted number is {number}")
      raise ValueError(f"Real number is {n}, extracted number is {number}")

    # article = {
    #   "url": response.url,
    #   "title": title,
    #   "category": category,
    #   "topics": topics,
    #   "date_publishing_resistenze": date_publishing_resistenze,
    #   "number": n, #number,
    #   "author": author,
    #   "source": source,
    #   "date_publishing_source": date_publishing_source,
    #   "translator": translator,
    #   "contents": text,
    # }

    return article

  def html_to_text(self, html, tag_name = None) -> str:
    soup = BeautifulSoup(html, "html5lib") # Use html5lib parser for better compatibility with HTML5
    tag = soup.find(tag_name) if tag_name else soup

    if tag is None:
      return ""

    for br in tag.find_all("br"): # Replace <br> tags with \n
      br.replace_with("\n")

    text = tag.get_text(separator = "\n", strip = True)
    return text

  def articles_load(self) -> None:
    """Load articles data."""
    self.logger.info(f"Loading articles")

    if not os.path.exists(Config.ARTICLES_PATH):
      raise ValueError(f"Articles not found at {Config.ARTICLES_PATH}")

    with open(Config.ARTICLES_PATH, 'rb') as f:
      self._articles = pickle.load(f)

  def data_sync(self) -> int:
    """Sync data: faiss index, metadata and texts."""
    embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

    # Initialize or load existing data
    if all(os.path.exists(path) for path in [
      Config.FAISS_INDEX_PATH,
      Config.METADATA_PATH,
      Config.TEXTS_PATH
    ]):
      self.logger.info("Syncing existing data from disk...")
      index = faiss.read_index(Config.FAISS_INDEX_PATH)
      with open(Config.METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
      with open(Config.TEXTS_PATH, 'rb') as f:
        texts = pickle.load(f)
    else:
      self.logger.info("Syncing new data...")
      metadata = []
      texts = []
      dim = embedding_model.get_sentence_embedding_dimension()
      index = faiss.IndexFlatL2(dim)

    # Track new data
    new_articles = 0
    new_embeddings = []
    new_texts = []
    new_metadata = []

    total = Config.ARTICLES_MAX if Config.ARTICLES_MAX > 0 else len(self._articles)
    finished = False
    with tqdm(total = total, desc = "Indexing articles") as pbar:
      for n, article in enumerate(self._articles):
        try:
          chunks = self.chunk_text(article['contents'])
          filename = os.path.basename(article.get('__filepath'))
          new_article = False

          for chunk in chunks:
            # Only process chunks we haven't seen before in texts chunks
            if chunk not in texts:
              new_article = True
              new_texts.append(chunk)
              new_metadata.append({
                'url': article.get('url', ''),
                'title': article.get('title', ''),
                #'author': article.get('author', ''),
                #'date': article.get('date_publishing_resistenze', ''),
                #'source_path': article['__filepath'],
                'last_updated': article['__mtime'].isoformat()
              })
              new_embeddings.append(embedding_model.encode(chunk))
          if new_article:
            new_articles += 1
            pbar.set_description(f"Added chunks from article {1+n}")
          pbar.update(1)
          if Config.ARTICLES_MAX > 0 and n >= Config.ARTICLES_MAX:
            self.logger.info(f"Stopped indexing after {Config.ARTICLES_MAX} articles.")
            finished = True
            break

        except Exception as e:
          finished = True
          self.logger.error(f"Error chunking article number {article["number"]} in file {article.get('__filepath')}: {e}", exc_info = self.log_level <= logging.DEBUG)
          raise
        finally:
          if n >= len(self._articles) or finished:
            pbar.close()
          
    if new_embeddings:
      # Add new data to existing structures
      texts.extend(new_texts)
      metadata.extend(new_metadata)
      embeddings_array = np.array(new_embeddings).astype('float32')
      index.add(embeddings_array)

      faiss.write_index(index, Config.FAISS_INDEX_PATH) # Save faiss index
      with open(Config.METADATA_PATH, 'wb') as f: # Save metadata
        pickle.dump(metadata, f)
      with open(Config.TEXTS_PATH, 'wb') as f: # Save texts
        pickle.dump(texts, f)

      self.logger.info(f"Indexed {len(new_embeddings)} new chunks from {new_articles} articles.")
    else:
      self.logger.info("No new articles to index")
    return len(new_embeddings)

  def chunk_text(self, text, size = Config.CHUNK_SIZE, overlap = Config.CHUNK_OVERLAP) -> List:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
      chunk = ' '.join(words[i:i+size])
      if chunk:
        chunks.append(chunk)
    return chunks

  def download_and_sync(self) -> int:
    """execute the download and sync process"""
    self.logger.info("Starting article download processes")
    self.articles_load()
    count = self.articles_download()
    self.data_sync()
    self.logger.info(f"Successfully processed {count} articles")
    return count

def get_article_manager(log_level) -> ArticleManager:
  """Factory function to get the article manager instance"""
  return ArticleManager(log_level)

if __name__ == "__main__":
  try:
    article_manager = get_article_manager(log_level = logging.INFO)
    count = article_manager.download_and_sync()
    article_manager.logger.info(f"{count} articles downloaded and synced")
    exit(0)
  except KeyboardInterrupt:
    article_manager.logger.warning('Interrupted')
    exit(130)
