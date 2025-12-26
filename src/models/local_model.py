import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from typing import Dict, Optional

from ..utils.logger import logger


class LocalModel:
    """Wrapper for local LLM with quantization"""
    
    def __init__(
        self,
        model_id: str,
        quantization: str = "4bit",
        max_tokens: int = 256,  # Reduced for CPU inference - 2048 was too slow
        temperature: float = 0.7
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        logger.info(f"Loading {model_id} with {quantization} quantization...")
        
        # Configure 4-bit quantization
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization else "auto"
        )
        
        self.device = self.model.device
        
        logger.info(f"✓ Model loaded on {self.device}")
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Generate response from prompt
        
        Args:
            prompt: User query
            context: Optional context from RAG
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with text, input_tokens, output_tokens
        """
        
        max_new_tokens = max_tokens or self.max_tokens
        temp = temperature or self.temperature
        
        # Build full prompt with chat template for TinyLlama/Qwen models
        if context:
            system_msg = "You are a helpful AI assistant. Use the provided context to answer the question accurately and concisely."
            user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            system_msg = "You are a helpful AI assistant. Answer questions accurately and concisely."
            user_msg = prompt
        
        # Use chat template format (TinyLlama uses ChatML format)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        # Apply chat template if available, otherwise use fallback format
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            full_prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            output_length = outputs.shape[1] - input_length
            
            return {
                'text': response.strip(),
                'input_tokens': input_length,
                'output_tokens': output_length
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text, truncation=True, max_length=4096))
    
    def get_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free"""
        return 0.0
    
    def get_info(self) -> Dict:
        """Get model information"""
        return {
            'model_id': self.model_id,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'device': str(self.device),
            'cost': 0.0
        }