# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from typing import Any, Optional, List, Dict
import base64
import os

def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def send_generate_request(
    messages,
    model_path="/data/public/Qwen3-VL-8B-Instruct",  # Updated to use local model path
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_tokens=4096,
    quantization=True,  # Add quantization option
):
    """
    Uses a local Qwen model to generate response based on messages
    Replaces the old API-based approach with local model inference
    
    Args:
        messages: A list of message dicts, each containing role and content
        model_path: Path to the local Qwen model
        device: Device to run the model on
        max_tokens: Maximum number of tokens to generate
        quantization: Whether to use 8-bit quantization to reduce memory usage
    
    Returns:
        str: The generated response text from the local model
    """
    # Initialize model and tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Use quantization to reduce memory usage if requested
    if quantization and device.startswith('cuda'):
        try:
            import bitsandbytes
            # Use 8-bit quantization to reduce memory usage
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use float16 to reduce memory
                trust_remote_code=True,
                device_map=device,
                load_in_8bit=True,  # Enable 8-bit quantization
            ).eval()
            print("Model loaded with 8-bit quantization for reduced memory usage")
        except ImportError:
            print("bitsandbytes not available, loading normally")
            # Fall back to normal loading if bitsandbytes is not available
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use float16 to reduce memory
                trust_remote_code=True,
                device_map=device
            ).eval()
    else:
        # Load normally without quantization
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
            trust_remote_code=True,
            device_map=device
        ).eval()
    
    # Construct the full prompt from messages
    full_prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if isinstance(content, list):
            # Handle list of content items (text and images)
            text_content = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_content = item["text"]
                    elif item.get("type") == "image":
                        # For now, we'll just note that there's an image but won't process it in this simplified version
                        if text_content:
                            text_content += " [IMAGE]"
                        else:
                            text_content = "[IMAGE]"
                else:
                    text_content = str(content)
            content = text_content
        elif not isinstance(content, str):
            content = str(content)
            
        if role == "user":
            full_prompt += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            full_prompt += f"<|assistant|>\n{content}</s>\n"
        elif role == "system":
            full_prompt += f"<|system|>\n{content}</s>\n"
    
    # Tokenize the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id  # Prevent warning
        )
    
    # Clear GPU cache to free memory
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # Decode the response - only get the newly generated part
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response if response else None


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    This function is kept for compatibility but won't be used with local models
    """
    print("send_direct_request is not used with local model approach")
    return None