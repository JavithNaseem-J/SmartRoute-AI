import openai
import anthropic
from typing import Dict, List, Optional
import tiktoken
from .base_model import BaseModel

class APIModel(BaseModel):
    """Wrapper for API-based models (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.provider = config['provider']
        self.api_key = self._get_api_key()
        self.client = self._init_client()
        self.encoder = self._init_tokenizer()
        
    def _get_api_key(self) -> str:
        """Retrieve API key from environment"""
        import os
        key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        key = os.getenv(key_map.get(self.provider))
        if not key:
            raise ValueError(f"API key not found for {self.provider}")
        return key
    
    def _init_client(self):
        """Initialize the appropriate API client"""
        if self.provider == 'openai':
            return openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            return anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting"""
        if self.provider == 'openai':
            try:
                return tiktoken.encoding_for_model(self.name)
            except:
                return tiktoken.get_encoding("cl100k_base")
        else:
            # Anthropic uses similar tokenization
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate response from API"""
        start_time = time.time()
        
        try:
            if self.provider == 'openai':
                response = self._openai_generate(prompt, **kwargs)
            elif self.provider == 'anthropic':
                response = self._anthropic_generate(prompt, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                'response': response['text'],
                'tokens_used': response['tokens'],
                'cost': self.calculate_cost(response['tokens']),
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
    
    def _openai_generate(self, prompt: str, **kwargs) -> Dict:
        """Generate using OpenAI API"""
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)
        
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        input_tokens = self.count_tokens(prompt)
        output_tokens = response.usage.completion_tokens
        
        return {
            'text': response.choices[0].message.content,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            }
        }
    
    def _anthropic_generate(self, prompt: str, **kwargs) -> Dict:
        """Generate using Anthropic API"""
        max_tokens = kwargs.get('max_tokens', 1000)
        temperature = kwargs.get('temperature', 0.7)
        
        response = self.client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        return {
            'text': response.content[0].text,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            }
        }
    
    def calculate_cost(self, tokens: Dict) -> float:
        """Calculate cost based on token usage"""
        input_cost = (tokens['input'] / 1000) * self.config['cost_per_1k_tokens']['input']
        output_cost = (tokens['output'] / 1000) * self.config['cost_per_1k_tokens']['output']
        return input_cost + output_cost
    
    async def agenerate(self, prompt: str, **kwargs) -> Dict:
        """Async version of generate"""
        # Implement async version if needed
        return self.generate(prompt, **kwargs)