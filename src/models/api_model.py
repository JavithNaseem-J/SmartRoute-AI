import os
import time
import tiktoken
from typing import Dict, Optional, Literal
from openai import OpenAI, RateLimitError
from anthropic import Anthropic

from ..utils.logger import logger


class APIModel:
    """Wrapper for API-based LLMs (OpenAI, Anthropic)"""
    
    def __init__(
        self,
        model_id: str,
        provider: Literal["openai", "anthropic"],
        cost_per_1k_input: float,
        cost_per_1k_output: float,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        self.model_id = model_id
        self.provider = provider
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize client
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            self.client = OpenAI(api_key=api_key)
            
            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_id)
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
            self.client = Anthropic(api_key=api_key)
            self.tokenizer = None  # Anthropic handles tokenization internally
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"✓ Initialized {provider} client for {model_id}")
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Generate response with retry logic
        
        Args:
            prompt: User query
            context: Optional context from RAG
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with text, input_tokens, output_tokens
        """
        
        max_tok = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        
        # Build full prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = prompt
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._generate_openai(full_prompt, max_tok, temp)
                else:
                    return self._generate_anthropic(full_prompt, max_tok, temp)
            
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded: {e}")
                    raise
            
            except Exception as e:
                logger.error(f"Generation failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
    
    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """OpenAI-specific generation"""
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            'text': response.choices[0].message.content,
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens
        }
    
    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict:
        """Anthropic-specific generation"""
        
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            'text': response.content[0].text,
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimate for Anthropic (4 chars per token)
            return len(text) // 4
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate actual cost based on token usage"""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_id': self.model_id,
            'provider': self.provider,
            'cost_per_1k_input': self.cost_per_1k_input,
            'cost_per_1k_output': self.cost_per_1k_output,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }