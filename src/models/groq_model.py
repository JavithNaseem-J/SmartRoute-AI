import os
import time
from typing import Dict, List, Optional
from groq import Groq, RateLimitError

from src.utils.logger import logger


class GroqModel:

    def __init__(
        self,
        model_id: str,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        max_retries: int = 3
    ):
        self.model_id = model_id
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Get your FREE API key at: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=api_key)
        
        logger.info(f"####### Initialized Groq client for {model_id} #######")
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict:
        max_tok = self.max_tokens
        temp = self.temperature

        # Build messages: system → history → current user turn.
        # History is a list of {"role": "user"|"assistant", "content": ...} dicts
        # from previous turns. Inserting it here gives the model full context
        # of the conversation without changing any other part of the pipeline.
        if context:
            system_msg = "You are a helpful AI assistant. Use the provided context to answer the question accurately and concisely."
            user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            system_msg = "You are a helpful AI assistant. Answer questions accurately and concisely."
            user_msg = prompt

        messages = [
            {"role": "system", "content": system_msg},
            *(history or []),          # previous turns, empty list if no session
            {"role": "user", "content": user_msg}
        ]
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tok,
                    temperature=temp
                )
                
                return {
                    'text': response.choices[0].message.content.strip(),
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens
                }
            
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} retries")
                    raise
            
            except Exception as e:
                logger.error(f"Groq API error: {e}")
                raise

    def generate_stream(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict]] = None,
    ):
        """Stream tokens from Groq using server-sent events."""
        max_tok = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        if context:
            system_msg = "You are a helpful AI assistant. Use the provided context to answer the question accurately and concisely."
            user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            system_msg = "You are a helpful AI assistant. Answer questions accurately and concisely."
            user_msg = prompt

        messages = [
            {"role": "system", "content": system_msg},
            *(history or []),
            {"role": "user", "content": user_msg}
        ]

        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tok,
            temperature=temp,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accurate budget estimates."""
        try:
            import tiktoken
            # cl100k_base is OpenAI's encoding, but it's very close to Llama's
            # BPE tokenizer counts, and is extremely fast locally.
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            logger.warning("tiktoken not installed, falling back to heuristic token count")
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
            'provider': 'groq',
            'cost_per_1k_input': self.cost_per_1k_input,
            'cost_per_1k_output': self.cost_per_1k_output,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
