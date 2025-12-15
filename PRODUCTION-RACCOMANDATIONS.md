# Recommended Models for Production

For Italian Language Excellence
LLM Options (Best to Good):
1. Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct ⭐ RECOMMENDED
yamlmodels:
  llm:
    model_name: "Qwen/Qwen2.5-7B-Instruct"  # or Qwen2.5-14B-Instruct
    max_tokens: 1024
    temperature: 0.1
    device: "cuda"  # Will use GPU automatically

Why: Excellent multilingual support including Italian, very few typos
RAM needed: 7B needs ~14GB RAM, 14B needs ~28GB RAM
VRAM needed: 7B needs ~16GB GPU, 14B needs ~32GB GPU
Quality: Production-grade, minimal errors
Speed: Fast inference on GPU

2. Llama-3.1-8B-Instruct (Meta)
yamlmodel_name: "meta-llama/Llama-3.1-8B-Instruct"

Why: Excellent Italian support, Meta's latest
VRAM: ~16GB GPU
Quality: Very high, comparable to Qwen2.5-7B

3. Mistral-7B-Instruct-v0.3 (Mistral AI)
yamlmodel_name: "mistralai/Mistral-7B-Instruct-v0.3"

Why: Strong European language support
VRAM: ~14GB GPU
Quality: Good Italian, fewer typos than smaller models

4. Gemma-2-9B-it (Google)
yamlmodel_name: "google/gemma-2-9b-it"

Why: Specifically instruction-tuned, good multilingual
VRAM: ~18GB GPU

Embedding Options:
1. multilingual-e5-large ⭐ RECOMMENDED
yamlembedding:
  model_name: "intfloat/multilingual-e5-large"
  batch_size: 64
  dimension: 1024

Why: State-of-the-art multilingual embeddings
Quality: Superior semantic understanding for Italian
Size: ~2GB, very efficient

2. multilingual-e5-base (Good balance)
yamlembedding:
  model_name: "intfloat/multilingual-e5-base"
  dimension: 768

Why: Lighter than large, still excellent quality
Size: ~1GB

3. LaBSE (Language-agnostic BERT)
yamlembedding:
  model_name: "sentence-transformers/LaBSE"
  dimension: 768

Why: Specifically designed for cross-lingual tasks
Good for: Italian semantic search


Production Configuration Example
For Machine with RTX 4090 (24GB VRAM) or A100 (40GB)
yamldatabase:
  mongodb_uri: "mongodb://localhost:27017/"
  database_name: "political_journal"
  collection_name: "articles"

models:
  llm:
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    max_tokens: 1024
    temperature: 0.1
    device: "cuda"
    quantization: null  # or "4bit" to save VRAM
  
  embedding:
    model_name: "intfloat/multilingual-e5-large"
    batch_size: 64
    dimension: 1024

ingestion:
  articles_path: "./data/articles"
  metadata_path: "./data/metadata"
  batch_size: 32  # Larger batches with GPU
  generate_summaries: false

retrieval:
  max_results: 10
  min_similarity_score: 0.25
For Machine with RTX 3090 (24GB) or Similar
yamlmodels:
  llm:
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    quantization: "4bit"  # Reduces VRAM to ~7GB
  
  embedding:
    model_name: "intfloat/multilingual-e5-large"
For Machines with Limited VRAM (8-12GB)
yamlmodels:
  llm:
    model_name: "Qwen/Qwen2.5-3B-Instruct"  # Your current choice is good!
    device: "cuda"
  
  embedding:
    model_name: "intfloat/multilingual-e5-base"

Code Changes for Quantization (Optional)
If you want to use 4-bit quantization to fit larger models, update src/llm/query_classifier.py:
pythondef __init__(self, config_path: str = "config/config.yaml"):
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
    
    # Prepare loading arguments
    load_kwargs = {
        "low_cpu_mem_usage": True,
    }
    
    if use_quantization == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        load_kwargs["device_map"] = "auto"
    elif use_quantization == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
    
    # Load model and tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        **load_kwargs
    )
    
    print("✓ LLM model loaded")
For quantization, also install:
bashpip install bitsandbytes accelerate

Expected Quality Improvements
Model SizeItalian TyposSemantic QualitySpeed (GPU)0.5BFrequentBasicVery Fast3BOccasionalGoodFast7BRareExcellentMedium14BVery RareOutstandingSlower

My Top Recommendation
For production with good GPU:

LLM: Qwen/Qwen2.5-7B-Instruct (best balance)
Embeddings: intfloat/multilingual-e5-large

This combination will give you:

✅ Near-perfect Italian grammar
✅ Excellent semantic understanding
✅ Reliable query classification
✅ Professional-quality answers
✅ Reasonable resource usage

The jump from 0.5B to 7B is dramatic for language quality, while still being manageable on modern GPUs.
