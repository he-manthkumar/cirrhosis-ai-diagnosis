"""
LLM Service - Integrates with LLM for narrative generation.

This service handles communication with OpenAI API
to generate human-readable clinical narratives from model explanations.
Supports optional image input for external symptom assessment.
"""
import os
import base64
from typing import Optional
from dotenv import load_dotenv
from backend.utils.image_utils import validate_base64_image, get_image_info

load_dotenv()


class LLMService:
    """
    Service for generating clinical narratives using OpenAI.
    
    Features:
    - Text-based narrative generation from model explanations
    - Optional image analysis for external symptoms (jaundice, spider angiomas, etc.)
    - Image findings are given lower weightage compared to clinical data
    """
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
    async def generate_narrative(
        self, 
        prompt: str, 
        image_base64: Optional[str] = None,
        image_mime_type: str = "image/jpeg"
    ) -> str:
        """
        Generate a clinical narrative from the explanation prompt.
        
        Args:
            prompt: The formatted prompt with rules and patient context
            image_base64: Optional base64-encoded image of external symptoms
            image_mime_type: MIME type of the image (default: image/jpeg)
            
        Returns:
            Generated narrative string
        """
        if self.provider == "openai":
            return await self._generate_openai(prompt, image_base64, image_mime_type)
        else:
            return self._generate_fallback(prompt)
    
    async def _generate_openai(
        self, 
        prompt: str, 
        image_base64: Optional[str] = None,
        image_mime_type: str = "image/jpeg"
    ) -> str:
        """Generate using OpenAI API with optional image input."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.openai_api_key)
            
            # System message for medical context
            system_message = {
                "role": "system",
                "content": "You are a medical AI assistant generating clinical risk explanations. "
                            "IMPORTANT RULES:\n"
                            "- The patient is currently ALIVE.\n"
                            "- Do NOT state or imply that the patient is dead.\n"
                            "- Frame all explanations in terms of MORTALITY RISK and DISEASE SEVERITY.\n"
                            "- Use phrases such as 'high risk of death without intervention' instead of 'death'.\n"
                            "- The output is a prognostic explanation, NOT an outcome declaration.\n"
                            "- Maintain a calm, professional, clinician-facing tone."
            }
            
            # Build the user message content
            if image_base64:
                # Validate image before sending
                if not validate_base64_image(image_base64):
                    print("Warning: Invalid base64 image, proceeding without image")
                    image_base64 = None
            
            if image_base64:
                # Add image context with lower weightage instruction
                image_prompt = self._build_image_prompt(prompt)
                
                # Create content with image for vision model
                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{image_base64}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "text",
                        "text": image_prompt
                    }
                ]
                
                # Use vision-capable model
                model = "gpt-4o-mini"
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        system_message,
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=400,
                    temperature=0.3
                )
            else:
                # Text-only generation
                response = await client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        system_message,
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
            
            return response.choices[0].message.content
            
        except ImportError:
            print("openai package not installed. Run: pip install openai")
            return self._generate_fallback(prompt)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback(prompt)
    
    def _build_image_prompt(self, base_prompt: str) -> str:
        """
        Build a prompt that includes image analysis with lower weightage.
        
        The image shows external symptoms that may be relevant but should
        not override the clinical data and model predictions.
        """
        image_analysis_instruction = """

---
ADDITIONAL CONTEXT - EXTERNAL SYMPTOM IMAGE (LOWER WEIGHTAGE):
An image showing the patient's external symptoms has been provided. Please analyze this image for any visible signs that may be relevant to liver cirrhosis assessment, such as:
- Jaundice (yellowing of skin or eyes)
- Spider angiomas (spider-like blood vessels on skin)
- Palmar erythema (reddening of palms)
- Ascites (visible abdominal distension)
- Edema (swelling in extremities)
- Bruising or skin changes

IMPORTANT WEIGHTING INSTRUCTIONS:
1. The clinical lab values and model predictions should be given PRIMARY importance (approximately 80% weight)
2. The image findings should be considered as SUPPLEMENTARY evidence only (approximately 20% weight)
3. If image findings contradict clinical data, prioritize the clinical data but note the discrepancy
4. Include a brief mention of any relevant image findings at the end of your narrative
5. If no relevant signs are visible in the image, simply state that no additional external symptoms were observed

Please integrate any relevant image observations into your clinical narrative, ensuring they complement but do not override the primary analysis based on clinical data.
"""
        return base_prompt + image_analysis_instruction
    
    def _generate_fallback(self, prompt: str) -> str:
        """
        Fallback template-based narrative when LLM is unavailable.
        Extracts key information from the prompt and creates a basic narrative.
        """
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
            return bool(self.openai_api_key)
        return False
    
    async def analyze_image_only(
        self,
        image_base64: str,
        image_mime_type: str = "image/jpeg"
    ) -> dict:
        """
        Analyze an external symptom image independently.
        
        This can be used to get image-based findings before combining
        with clinical data predictions.
        
        Args:
            image_base64: Base64-encoded image
            image_mime_type: MIME type of the image
            
        Returns:
            Dictionary with image analysis findings
        """
        try:
            # Validate image first
            if not validate_base64_image(image_base64):
                image_info = get_image_info(image_base64)
                return {
                    "success": False,
                    "error": "Invalid or corrupted base64 image. Please provide a complete, valid image.",
                    "analysis": None,
                    "debug_info": image_info
                }
            
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.openai_api_key)
            
            analysis_prompt = """Analyze this medical image for signs that may be relevant to liver disease assessment.

Look for and report on the following (if visible):
1. Jaundice (yellowing of skin or sclera)
2. Spider angiomas (spider-like blood vessels)
3. Palmar erythema (red palms)
4. Visible abdominal distension (ascites)
5. Peripheral edema (swelling)
6. Bruising or petechiae
7. Muscle wasting
8. Any other relevant clinical signs

For each finding, indicate:
- Whether it is present/absent/not visible in image
- Severity if present (mild/moderate/severe)
- Confidence level (low/medium/high)

Respond in a structured format. If the image quality is poor or the relevant body part is not visible, indicate that."""

            # Create content with image
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{image_base64}",
                        "detail": "low"
                    }
                },
                {
                    "type": "text",
                    "text": analysis_prompt
                }
            ]
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical image analysis assistant. Analyze images for clinical signs relevant to liver disease."
                    },
                    {"role": "user", "content": user_content}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "note": "Image findings should be weighted lower than clinical data (approximately 20% weight)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
