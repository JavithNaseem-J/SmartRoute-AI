import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import time
from .base_model import BaseModel

class LocalModel(BaseModel):
    """Wrapper for locally hosted models"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
    def load(self):
        """Lazy load the model"""
        if self.loaded:
            return
            
        print(f"Loading {self.name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.loaded = True
        print(f"{self.name} loaded successfully")
    
    def unload(self):
        """Unload model to free memory"""
        if self.loaded:
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.loaded = False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not self.loaded:
            self.load()
        return len(self.tokenizer.encode(text))
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate response locally"""
        if not self.loaded:
            self.load()
        
        start_time = time.time()
        
        try:
            max_tokens = kwargs.get('max_tokens', 1000)
            temperature = kwargs.get('temperature', 0.7)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response_text = self.tokenizer.decode(
                outputs[0][input_tokens:], 
                skip_special_tokens=True
            )
            
            output_tokens = outputs.shape[1] - input_tokens
            latency = (time.time() - start_time) * 1000
            
            tokens = {
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            }
            
            return {
                'response': response_text,
                'tokens_used': tokens,
                'cost': self.calculate_cost(tokens),
                'latency_ms': latency,
                'model': self.name,
                'success': True
            }
            
        except Exception as e:
            return {
                'response': None,
                'error': str(e),
                'success': False,
                'model': self.name
            }
    
    def calculate_cost(self, tokens: Dict) -> float:
        """Calculate cost (minimal for local models)"""

        total_tokens = tokens['total']
        cost_per_1k = self.config['cost_per_1k_tokens']['input']
        return (total_tokens / 1000) * cost_per_1k
    
    async def agenerate(self, prompt: str, **kwargs) -> Dict:
        """Async version - local models are synchronous"""
        return self.generate(prompt, **kwargs)