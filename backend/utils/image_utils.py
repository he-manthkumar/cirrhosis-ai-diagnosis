"""
Image Utilities - Helper functions for image processing.

Provides utilities for:
- Converting images to base64 for API requests
- Validating image formats
- Handling image data for OpenAI API
"""
import base64
import os
from typing import Optional, Tuple


def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 encoded string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a supported image format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported image format: {ext}. Supported: {valid_extensions}")
    
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    
    return encoded


def base64_to_bytes(base64_string: str) -> bytes:
    """
    Convert base64 string back to bytes.
    
    Args:
        base64_string: Base64 encoded string
        
    Returns:
        Decoded bytes
    """
    return base64.b64decode(base64_string)


def get_mime_type(image_path: str) -> str:
    """
    Get MIME type from image file extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (e.g., 'image/jpeg')
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')


def prepare_image_for_api(image_path: str) -> Tuple[str, str]:
    """
    Prepare an image for API request by converting to base64 and getting MIME type.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_string, mime_type)
    """
    base64_string = image_to_base64(image_path)
    mime_type = get_mime_type(image_path)
    return base64_string, mime_type


def validate_base64_image(base64_string: str) -> bool:
    """
    Validate if a string is a valid base64 encoded image.
    
    Args:
        base64_string: The base64 string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if it can be decoded
        decoded = base64.b64decode(base64_string)
        
        # Check for common image headers (magic bytes)
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        # GIF: 47 49 46 38
        # WebP: 52 49 46 46
        
        if len(decoded) < 4:
            return False
        
        jpeg_header = decoded[:3] == b'\xff\xd8\xff'
        png_header = decoded[:4] == b'\x89PNG'
        gif_header = decoded[:4] == b'GIF8'
        webp_header = decoded[:4] == b'RIFF'
        
        return jpeg_header or png_header or gif_header or webp_header
        
    except Exception:
        return False


def get_image_info(base64_string: str) -> Optional[dict]:
    """
    Get information about a base64 encoded image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Dictionary with image info or None if invalid
    """
    try:
        decoded = base64.b64decode(base64_string)
        
        # Detect format from magic bytes
        if decoded[:3] == b'\xff\xd8\xff':
            format_name = 'JPEG'
            mime_type = 'image/jpeg'
        elif decoded[:4] == b'\x89PNG':
            format_name = 'PNG'
            mime_type = 'image/png'
        elif decoded[:4] == b'GIF8':
            format_name = 'GIF'
            mime_type = 'image/gif'
        elif decoded[:4] == b'RIFF':
            format_name = 'WebP'
            mime_type = 'image/webp'
        else:
            return None
        
        return {
            'format': format_name,
            'mime_type': mime_type,
            'size_bytes': len(decoded),
            'base64_length': len(base64_string)
        }
        
    except Exception:
        return None
