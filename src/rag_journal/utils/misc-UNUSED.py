import os
from pathlib import Path
from rag_journal.utils.config import CONFIG

class Utils:
  """Main RAG system utilities"""

  def UNUSED_get_cached_model_path(self, model_name: str) -> str:
    if not CONFIG["models"]["local_cache_only"]:
      print(f"  Will attempt download (requires internet)")
      return model_name

    """Get the path to cached model, or return model_name if not cached"""
    cache_dir = os.environ.get(
      'HF_HOME', 
      os.path.join(Path.home(), '.cache', 'huggingface')
    )
    
    # Convert model name to cache format (e.g., "Qwen/Qwen2.5-3B-Instruct" -> "models--Qwen--Qwen2.5-3B-Instruct")
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, 'hub', model_cache_name)
    
    # Check if model exists in cache
    if os.path.exists(model_cache_path):
      snapshots_dir = os.path.join(model_cache_path, 'snapshots')
      if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
          full_path = os.path.join(snapshots_dir, snapshots[0])
          print(f"Si usa il modello in cache")
          #print(f"Using cached model from: {full_path}")
          return full_path
    
    # Not in cache - will require download
    print(f"  Model not in cache at {model_cache_path}")
    print(f"  Will attempt download (requires internet)")
    return model_name