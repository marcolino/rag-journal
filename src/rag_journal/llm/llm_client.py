import os
import sys
import json
from typing import Dict, Any, List #, Optional
from abc import ABC, abstractmethod
from rag_journal.utils.logger import logger
from rag_journal.utils.config import CONFIG


class LLMClient(ABC):
  """Base class for LLM clients"""
  
  @abstractmethod
  def generate_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
    """Generate response with tool calling support"""
    pass
  
  @abstractmethod
  def generate(self, prompt: str, max_tokens: int = None) -> str:
    """Simple text generation"""
    pass


class OpenAIClient(LLMClient):
  """OpenAI API client"""
  
  def __init__(self, config: Dict):
    #import openai
    from openai import OpenAI
    
    self.config = config
    models_config = config['models']
    api_config = models_config['api']

    # Get API key from environment
    api_key_env_var = api_config['api_key_env'] # "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env_var)
    
    if not api_key:
      raise ValueError(f"Missing {api_key_env_var} environment variable")
    # api_key = os.environ.get(config['api_key_env'])
    # if not api_key:
    #   raise ValueError(f"Missing {self.models_config['api']['api_key_env']} environment variable")
    
    #self.client = openai.OpenAI(api_key=api_key)
    self.client = OpenAI(api_key=api_key)
    self.model = models_config['api']['model']
    self.max_tokens = models_config['api'].get('max_tokens', 2000)
    self.temperature = models_config['api'].get('temperature', 0.1)
  
  def generate_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
    """Generate with function calling"""
    
    # Converti tools nel formato OpenAI
    openai_tools = [
      {
        "type": "function",
        "function": {
          "name": tool['name'],
          "description": tool['description'],
          "parameters": {
            "type": "object",
            "properties": tool['parameters'],
            "required": [k for k, v in tool['parameters'].items() 
              if not v.get('optional', False)]
          }
        }
      }
      for tool in tools
    ]
    
    try:
      response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=self.max_tokens,
        temperature=self.temperature
      )
    except Exception as e:
      logger.error(f"✗ Errore nella creazione di un completamento della chat: {e.message}")
      sys.exit(1)
    
    message = response.choices[0].message
    
    return {
      "content": message.content or "",
      "tool_calls": [
        {
          "id": tc.id,
          "name": tc.function.name,
          "parameters": json.loads(tc.function.arguments)
        }
        for tc in (message.tool_calls or [])
      ],
      "finish_reason": response.choices[0].finish_reason
    }
  
  def generate(self, prompt: str, max_tokens: int = None) -> str:
    """Simple generation"""
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens or self.max_tokens,
      temperature=self.temperature
    )
    return response.choices[0].message.content


class AnthropicClient(LLMClient):
  """Anthropic API client"""
  
  def __init__(self, config: Dict):
    import anthropic

    self.config = config
    models_config = config['models']
    api_config = models_config['api']

    # Get API key from environment
    api_key_env_var = api_config['api_key_env'] # "ANTHROPIC_API_KEY"
    api_key = os.environ.get(api_key_env_var)
    
    if not api_key:
      raise ValueError(f"Missing {api_key_env_var} environment variable")
    
    self.client = anthropic.Anthropic(api_key=api_key)
    self.model = self.models_config['api']['model']
    self.max_tokens = self.models_config['api'].get('max_tokens', 2000)
    self.temperature = self.models_config['api'].get('temperature', 0.1)
  
  def generate_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
    """Generate with tool use"""
    
    # Converti tools nel formato Anthropic
    anthropic_tools = [
      {
        "name": tool['name'],
        "description": tool['description'],
        "input_schema": {
          "type": "object",
          "properties": tool['parameters'],
          "required": [k for k, v in tool['parameters'].items() 
            if not v.get('optional', False)]
        }
      }
      for tool in tools
    ]
    
    # Estract system message if present
    system_msg = None
    user_messages = messages
    if messages and messages[0]['role'] == 'system':
      system_msg = messages[0]['content']
      user_messages = messages[1:]
    
    response = self.client.messages.create(
      model=self.model,
      system=system_msg,
      messages=user_messages,
      tools=anthropic_tools,
      max_tokens=self.max_tokens,
      temperature=self.temperature
    )
    
    # Estract tool calls
    tool_calls = []
    text_content = ""
    
    for block in response.content:
      if block.type == "text":
        text_content += block.text
      elif block.type == "tool_use":
        tool_calls.append({
          "id": block.id,
          "name": block.name,
          "parameters": block.input
        })
    
    return {
      "content": text_content,
      "tool_calls": tool_calls,
      "finish_reason": response.stop_reason
    }
  
  def generate(self, prompt: str, max_tokens: int = None) -> str:
    """Simple generation"""
    response = self.client.messages.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens or self.max_tokens,
      temperature=self.temperature
    )
    return response.content[0].text


class LocalLLMClient(LLMClient):
  """Local LLM client (requires a GPU)"""
  
  def __init__(self, config: Dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    self.model_name = config['llm_model']
    self.max_tokens = config.get('max_tokens', 2000)
    self.temperature = config.get('temperature', 0.1)
    
    logger.info(f"Si carica un LLM locale: {self.model_name}")
    
    model_path = self._get_cached_model_path(self.model_name)
    
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_path,
      dtype=torch.float16,
      device_map="auto",
      low_cpu_mem_usage=True
    )
    
    logger.info("✓ LLM locale caricato")
  
  def _get_cached_model_path(self, model_name: str) -> str:
    """Get cached model path"""
    from pathlib import Path
    cache_dir = os.environ.get('HF_HOME', 
                   os.path.join(Path.home(), '.cache', 'huggingface'))
    
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = os.path.join(cache_dir, 'hub', model_cache_name)
    
    if os.path.exists(model_cache_path):
      snapshots_dir = os.path.join(model_cache_path, 'snapshots')
      if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
          return os.path.join(snapshots_dir, snapshots[0])
    
    return model_name
  
  def generate_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
    """Generate with tool calling (Qwen 2.5+ format)"""
    
    # Qwen supporta tool calling con formato speciale
    prompt = self._format_tool_prompt(messages, tools)
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
    import torch
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=self.max_tokens,
        temperature=self.temperature,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
      )
    
    response = self.tokenizer.decode(
      outputs[0][inputs['input_ids'].shape[1]:],
      skip_special_tokens=True
    )
    
    # Parse tool calls from response
    tool_calls = self._parse_tool_calls(response)
    
    return {
      "content": response,
      "tool_calls": tool_calls,
      "finish_reason": "stop"
    }
  
  def _format_tool_prompt(self, messages: List[Dict], tools: List[Dict]) -> str:
    """Format prompt with tools for Qwen"""
    # Specific implementation for Qwen tool calling
    # See: https://github.com/QwenLM/Qwen2.5#tool-calling
    pass
  
  def _parse_tool_calls(self, response: str) -> List[Dict]:
    """Parse tool calls from model response"""
    # Parse JSON tool calls from response
    pass
  
  def generate(self, prompt: str, max_tokens: int = None) -> str:
    """Simple generation"""
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
    import torch
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_tokens or self.max_tokens,
        temperature=self.temperature,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
      )
    
    return self.tokenizer.decode(
      outputs[0][inputs['input_ids'].shape[1]:],
      skip_special_tokens=True
    )


def create_llm_client(config: Dict) -> LLMClient: #config_path: str = "config/config.yaml") -> LLMClient:
  """Factory function to create the appropriate LLM client"""
  
  # with open(self.models_config, 'r') as f:
  #   config = yaml.safe_load(f)
  
  mode = config['models']['mode']
  
  if mode == "api":
    api_config = config['models']['api']
    provider = api_config['provider']
    
    if provider == "openai":
      return OpenAIClient(config)
    elif provider == "anthropic":
      return AnthropicClient(config)
    else:
      raise ValueError(f"Unknown API provider: {provider}")
  
  elif mode == "local":
    return LocalLLMClient(config['models']['local'])
  
  else:
    raise ValueError(f"Unknown mode: {mode}. Use 'api' or 'local'")