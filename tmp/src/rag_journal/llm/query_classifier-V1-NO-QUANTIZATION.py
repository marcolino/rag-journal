import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import yaml
from typing import Dict, Any


class QueryClassifier:
  """LLM-based query classifier and parameter extractor"""
  
  def __init__(self, config_path: str = "config/config.yaml"):
    """Initialize query classifier"""
    with open(config_path, 'r') as f:
      config = yaml.safe_load(f)
    
    llm_config = config['models']['llm']
    
    self.model_name = llm_config['model_name']
    self.max_tokens = llm_config['max_tokens']
    self.temperature = llm_config['temperature']
    self.device = llm_config['device']
    use_quantization = llm_config.get('quantization', None)
    
    print(f"Loading LLM model: {self.model_name}")
    if use_quantization:
      print(f"  Using {use_quantization} quantization")
    print("⚠ This may take some time on first run...")
    
    # Load model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_name,
      #torch_dtype = torch.float32, # Use float32 for CPU
      dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
      #device_map = self.device,
      device_map = "auto" if torch.cuda.is_available() else None,
      low_cpu_mem_usage = True,
      #trust_remote_code = True, # If needed for custom models
    )
    
    print("✓ LLM model loaded")
    
    # Classification prompt template
    self.classification_prompt = """Analizza questa domanda in italiano e determina il tipo di query necessaria per un database di articoli giornalistici.

Domanda: {query}

Rispondi SOLO con un JSON valido in questo formato:
{{
  "query_type": "metadata" | "semantic" | "hybrid" | "analytical",
  "requires_count": true/false,
  "filters": {{
  "author": "nome autore o null",
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} o null,
  "categories": ["categoria1"] o null
  }},
  "semantic_query": "query per ricerca semantica o null",
  "reasoning": "breve spiegazione"
}}

Tipi di query:
- "metadata": solo filtri su metadati (autore, data, categoria)
- "semantic": ricerca semantica sul contenuto
- "hybrid": combinazione di filtri metadati + ricerca semantica
- "analytical": analisi complessa, trend temporali, confronti

Esempi:
Input: "Quanti articoli di Mario Rossi nel 2023?"
Output: {{"query_type": "metadata", "requires_count": true, "filters": {{"author": "Mario Rossi", "date_range": {{"start": "2023-01-01", "end": "2023-12-31"}}, "categories": null}}, "semantic_query": null, "reasoning": "Query su metadati con conteggio"}}

Input: "Cosa dice Giulia Verdi sull'Ucraina?"
Output: {{"query_type": "hybrid", "requires_count": false, "filters": {{"author": "Giulia Verdi", "date_range": null, "categories": null}}, "semantic_query": "Ucraina guerra", "reasoning": "Filtro autore + ricerca semantica"}}

Input: "Quali articoli parlano di energia rinnovabile?"
Output: {{"query_type": "semantic", "requires_count": false, "filters": {{}}, "semantic_query": "energia rinnovabile", "reasoning": "Ricerca semantica pura"}}
"""
  
  def classify_query(self, user_query: str) -> Dict[str, Any]:
    """Classify user query and extract parameters"""
    
    # Format prompt
    prompt = self.classification_prompt.format(query=user_query)
    
    # Prepare input for the model
    messages = [{"role": "user", "content": prompt}]
    text = self.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )
    
    # Tokenize
    inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
    
    # Generate
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=self.max_tokens,
        temperature=self.temperature,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
      )
    
    # Decode response
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
      print("RESPONSE:", response)
      json_obj = self._extract_json(response)
      return json_obj
    except Exception as e:
      print(f"⚠ Error parsing classification: {e}")
      print(f"Raw response: {response}")
      # Fallback to semantic search
      return {
        "query_type": "semantic",
        "requires_count": False,
        "filters": {},
        "semantic_query": user_query,
        "reasoning": f"Parsing failed, defaulting to semantic search. Error: {str(e)}"
      }
  
  # def _extract_json(self, text: str) -> Dict[str, Any]:
  #   """Extract JSON from model response"""
  #   # Try to find JSON in the response
  #   start_idx = text.find('{')
  #   end_idx = text.rfind('}') + 1
    
  #   if start_idx == -1 or end_idx == 0:
  #     raise ValueError("No JSON found in response")
    
  #   json_str = text[start_idx:end_idx]
  #   return json.loads(json_str)

  def _extract_json(self, text: str) -> Dict[str, Any]:
    """Extract JSON from model response with multiple fallback strategies"""
    import re
    
    text = text.strip()
    
    # Strategy 1: Remove markdown code blocks
    if '```' in text:
      # Match ```json ... ``` or ``` ... ```
      code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
      matches = re.findall(code_block_pattern, text, re.DOTALL)
      if matches:
        text = matches[0].strip()
    
    # Strategy 2: Extract JSON object
    # Find the outermost { } pair
    brace_count = 0
    start_idx = -1
    end_idx = -1
    
    for i, char in enumerate(text):
      if char == '{':
        if brace_count == 0:
          start_idx = i
        brace_count += 1
      elif char == '}':
        brace_count -= 1
        if brace_count == 0 and start_idx != -1:
          end_idx = i + 1
          break
    
    if start_idx == -1 or end_idx == -1:
      raise ValueError("No valid JSON object found in response")
    
    json_str = text[start_idx:end_idx].strip()
    
    # Strategy 3: Try to parse
    try:
      return json.loads(json_str)
    except json.JSONDecodeError as e:
      # Last resort: try to fix common issues
      # Remove trailing commas
      json_str = re.sub(r',\s*}', '}', json_str)
      json_str = re.sub(r',\s*]', ']', json_str)
      
      try:
        return json.loads(json_str)
      except json.JSONDecodeError:
        raise ValueError(f"Could not parse JSON: {e}\nExtracted text: {json_str}")
        
  def generate_answer(self, user_query: str, context: str) -> str:
    """Generate answer from retrieved articles"""
    
    prompt = f"""Basandoti SOLO sugli articoli seguenti, rispondi alla domanda dell'utente in modo conciso e accurato.

Articoli:
{context}

Domanda: {user_query}

Rispondi in italiano, in modo chiaro e conciso. Cita sempre le fonti (titolo e autore degli articoli).
Risposta:"""

    messages = [{"role": "user", "content": prompt}]
    text = self.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )
    
    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=3000)
    
    with torch.no_grad():
      outputs = self.model.generate(
        **inputs,
        max_new_tokens=self.max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id
      )
    
    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (everything after "Risposta:")
    if "Risposta:" in response:
      answer = response.split("Risposta:")[-1].strip()
    else:
      # Try to extract the last part
      answer = response.split("assistant")[-1].strip() if "assistant" in response else response
    
    return answer
