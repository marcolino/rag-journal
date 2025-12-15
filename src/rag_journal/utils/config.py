import yaml
from pathlib import Path

# Automatically load when module is imported
_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
with open(_CONFIG_PATH, 'r') as f:
  CONFIG = yaml.safe_load(f)
