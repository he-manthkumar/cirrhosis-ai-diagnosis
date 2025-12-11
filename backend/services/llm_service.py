"""
LLM Service - Integrates with LLM for narrative generation.

This service handles communication with Google Gemini API
to generate human-readable clinical narratives from model explanations.
Supports optional image input for external symptom assessment.
"""
import os
import base64
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """
    Service for generating clinical narratives using Google Gemini.
    
    Features:
    - Text-based narrative generation from model explanations
    - Optional image analysis for external symptoms (jaundice, spider angiomas, etc.)
    - Image findings are given lower weightage compared to clinical data
    """
    
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
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
        if self.provider == "gemini":
            return await self._generate_gemini(prompt, image_base64, image_mime_type)
        else:
            return self._generate_fallback(prompt)
    
    async def _generate_gemini(
        self, 
        prompt: str, 
        image_base64: Optional[str] = None,
        image_mime_type: str = "image/jpeg"
    ) -> str:
        """Generate using Google Gemini API with optional image input."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            
            # Use vision model if image is provided
            model_name = self.gemini_model
            if image_base64:
                # Ensure we use a vision-capable model
                if "flash" in model_name or "pro" in model_name:
                    pass  # These models support vision
                else:
                    model_name = "gemini-1.5-flash"
            
            model = genai.GenerativeModel(model_name)
            
            # Build the content parts
            if image_base64:
                # Add image context with lower weightage instruction
                image_prompt = self._build_image_prompt(prompt)
                
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                
                # Create content with image
                content = [
                    {
                        "mime_type": image_mime_type,
                        "data": image_data
                    },
                    image_prompt
                ]
                
                response = await model.generate_content_async(
                    content,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=400,
                        temperature=0.3
                    )
                )
            else:
                # Text-only generation
                system_instruction = (
                    "You are a medical AI assistant that explains machine learning "
                    "predictions to clinicians in clear, professional language."
                )
                
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=system_instruction
                )
                
                response = await model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=300,
                        temperature=0.3
                    )
                )
            
            return response.text
            
        except ImportError:
            print("google-generativeai package not installed. Run: pip install google-generativeai")
            return self._generate_fallback(prompt)
        except Exception as e:
            print(f"Gemini API error: {e}")
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
        if self.provider == "gemini":
            return bool(self.gemini_api_key)
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
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel(self.gemini_model)
            
            image_data = base64.b64decode(image_base64)
            
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

            content = [
                {
                    "mime_type": image_mime_type,
                    "data": image_data
                },
                analysis_prompt
            ]
            
            response = await model.generate_content_async(
                content,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.2
                )
            )
            
            return {
                "success": True,
                "analysis": response.text,
                "note": "Image findings should be weighted lower than clinical data (approximately 20% weight)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
