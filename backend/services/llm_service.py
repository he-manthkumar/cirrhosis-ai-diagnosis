"""
LLM Service - Integrates with LLM for narrative generation.

This service handles communication with an LLM (OpenAI, Ollama, etc.)
to generate human-readable clinical narratives from model explanations.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """
    Service for generating clinical narratives using an LLM.
    
    Supports multiple backends:
    - OpenAI API
    - Ollama (local)
    - Hugging Face (optional)
    """
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
    async def generate_narrative(self, prompt: str) -> str:
        """
        Generate a clinical narrative from the explanation prompt.
        
        Args:
            prompt: The formatted prompt with rules and patient context
            
        Returns:
            Generated narrative string
        """
        if self.provider == "openai":
            return await self._generate_openai(prompt)
        elif self.provider == "ollama":
            return await self._generate_ollama(prompt)
        else:
            return self._generate_fallback(prompt)
    
    async def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for this task
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that explains machine learning predictions to clinicians in clear, professional language."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for more consistent outputs
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            return self._generate_fallback(prompt)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback(prompt)
    
    async def _generate_ollama(self, prompt: str) -> str:
        """Generate using local Ollama instance."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "llama3.2",  # or mistral, etc.
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 300
                        }
                    },
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    return self._generate_fallback(prompt)
                    
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: str) -> str:
        """
        Fallback template-based narrative when LLM is unavailable.
        Extracts key information from the prompt and creates a basic narrative.
        """
        # This is a simple template-based fallback
        # In production, you'd want a more sophisticated approach
        
        narrative = (
            "The model's prediction was based on analysis of the patient's clinical values "
            "through an interpretable decision tree. The key factors that influenced this "
            "prediction include the patient's liver function markers and clinical presentation. "
            "Please refer to the extracted decision rules above for specific thresholds used. "
            "(Note: LLM service unavailable - using template response)"
        )
        
        return narrative
    
    def is_configured(self) -> bool:
        """Check if LLM service is properly configured."""
        if self.provider == "openai":
            return bool(self.api_key)
        elif self.provider == "ollama":
            return True  # Ollama runs locally
        return False
