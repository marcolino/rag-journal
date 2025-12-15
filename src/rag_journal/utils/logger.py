import logging
import sys
from pathlib import Path

def setup_logger(
  name: str = "rag_journal",
  log_file: str = "logs/rag.log",
  level: str = "INFO"
):
  """
  Simple one-level logger setup
  
  Args:
    name: Logger name
    log_file: Path to log file
    level: Log level (DEBUG, INFO, WARNING, ERROR)
  """
  # Create logs directory
  log_path = Path(log_file)
  log_path.parent.mkdir(exist_ok=True)
  
  # Get logger
  logger = logging.getLogger(name)
  logger.setLevel(getattr(logging, level.upper()))
  
  # Clear existing handlers
  logger.handlers.clear()
  
  # Console handler
  console_handler = logging.StreamHandler(sys.stdout)
  console_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
  )
  console_handler.setFormatter(console_format)
  logger.addHandler(console_handler)
  
  # File handler
  file_handler = logging.FileHandler(log_file, encoding='utf-8')
  file_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
  )
  file_handler.setFormatter(file_format)
  logger.addHandler(file_handler)
  
  return logger

# Global logger instance
logger = logging.getLogger("rag_journal") 