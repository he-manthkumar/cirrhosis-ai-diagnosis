import asyncio
import json
import os
from dotenv import load_dotenv

# Import our utility
from backend.utils.image_utils import (
    image_to_base64, 
    get_mime_type, 
    prepare_image_for_api,
    validate_base64_image,
    get_image_info
)

load_dotenv()


async def test_openai():
    """Test the OpenAI API with our LLM service"""
    from openai import AsyncOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    print(f"API Key: {api_key[:15]}..." if api_key else "API Key: NOT SET")
    print(f"Model: {model_name}")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        return
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        
        print("\n--- Testing text generation ---")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'Hello, Cirrhosis AI is working!' in one sentence."}],
            max_tokens=50
        )
        print(f"Response: {response.choices[0].message.content}")
        print("\n✅ OpenAI API is working!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


async def test_with_image():
    """Test OpenAI with image input using our utility"""
    from openai import AsyncOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4o-mini"  # Vision-capable model
    
    # Change this to your test image path
    image_path = r"C:\Users\Hemanth Kumar\OneDrive\Desktop\cirhossis.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    try:
        # Use our utility to prepare image
        image_base64, mime_type = prepare_image_for_api(image_path)
        
        print(f"\n--- Image Info ---")
        print(f"Path: {image_path}")
        print(f"MIME type: {mime_type}")
        print(f"Base64 length: {len(image_base64)} chars")
        
        # Validate the image
        is_valid = validate_base64_image(image_base64)
        print(f"Valid image: {is_valid}")
        
        info = get_image_info(image_base64)
        print(f"Image info: {info}")
        
        client = AsyncOpenAI(api_key=api_key)
        
        print("\n--- Testing image analysis ---")
        
        # Use correct structure for OpenAI vision
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Briefly describe what you see in this image."
                        }
                    ]
                }
            ],
            max_tokens=200
        )
        print(f"Image Analysis: {response.choices[0].message.content}")
        print("\n✅ Image analysis working!")
        
    except Exception as e:
        print(f"\n❌ Image Error: {e}")


def generate_api_request():
    """Generate a complete API request JSON with a real image."""
    image_path = r"C:\Users\Hemanth Kumar\OneDrive\Desktop\cirhossis.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Use our utility
    image_base64, mime_type = prepare_image_for_api(image_path)
    
    # Validate
    if not validate_base64_image(image_base64):
        print("❌ Invalid image!")
        return
    
    request = {
        "patient": {
            "age": 21464,
            "albumin": 2.6,
            "alk_phos": 1718,
            "ascites": "Y",
            "bilirubin": 14.5,
            "cholesterol": 261,
            "copper": 156,
            "drug": "D-penicillamine",
            "edema": "Y",
            "hepatomegaly": "Y",
            "platelets": 190,
            "prothrombin": 12.2,
            "sex": "F",
            "sgot": 137.95,
            "spiders": "Y",
            "stage": 4,
            "tryglicerides": 172
        },
        "image": {
            "image_base64": image_base64,
            "image_mime_type": mime_type
        }
    }
    
    # Save to file for easy copying
    with open("test_request.json", "w") as f:
        json.dump(request, f, indent=2)
    
    print(f"✅ Saved complete request to test_request.json")
    print(f"   Image format: {get_image_info(image_base64)}")
    print(f"\nUse this file content in Swagger or Postman to test /predict/full endpoint")


if __name__ == "__main__":
    print("=" * 50)
    print("OPENAI API TEST")
    print("=" * 50)
    
    # Run text test
    asyncio.run(test_openai())
    
    # Test with image
    print("\n" + "=" * 50)
    print("IMAGE TEST")
    print("=" * 50)
    asyncio.run(test_with_image())
    
    # Generate API request file
    print("\n" + "=" * 50)
    print("GENERATING API REQUEST FILE")
    print("=" * 50)
    generate_api_request()
