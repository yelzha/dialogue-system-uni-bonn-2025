import base64
import json
import requests
from PIL import Image
from io import BytesIO
from config import OLLAMA_BASE_URL

# Instruction prompt, adapted from original
instruction = (
    "Extract and return all structured fields from this receipt or invoice image in this JSON format:\n"
    "{\n"
    "  \"invoice_number\": string,\n"
    "  \"check_number\": string,\n"
    "  \"po_number\": string,\n"
    "  \"vendor\": string,\n"
    "  \"vendor_address\": string,\n"
    "  \"customer_name\": string,\n"
    "  \"customer_address\": string,\n"
    "  \"date\": string,\n"
    "  \"due_date\": string,\n"
    "  \"payment_date\": string,\n"
    "  \"amount\": string,\n"
    "  \"subtotal\": string,\n"
    "  \"tax\": string,\n"
    "  \"discount\": string,\n"
    "  \"total\": string,\n"
    "  \"currency\": string,\n"
    "  \"payment_method\": string,\n"
    "  \"account_number\": string,\n"
    "  \"routing_number\": string,\n"
    "  \"bank_name\": string,\n"
    "  \"items\": [\n"
    "    {\"item\": string, \"qty\": string, \"price\": string, \"total\": string}\n"
    "  ],\n"
    "  \"document_type\": string,\n"
    "  \"notes\": string\n"
    "}\n"
    "Return only the JSON. No explanations."
)

def image_to_base64(image_path):
    """Convert an image to base64 string for Ollama."""
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def parse_image(image_path):
    """Call Qwen2.5-VL on Ollama with an image and extract structured fields."""
    image_b64 = image_to_base64(image_path)

    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": instruction,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 4096  # You can increase to 4096 if needed
        }
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        parsed = json.loads(result)
    except Exception as e:
        print(f"[ERROR] Failed to parse image: {e}")
        parsed = {}

    # Fallback: build full structure with default values if missing
    fields = {
        "invoice_number": "",
        "check_number": "",
        "po_number": "",
        "vendor": "",
        "vendor_address": "",
        "customer_name": "",
        "customer_address": "",
        "date": "",
        "due_date": "",
        "payment_date": "",
        "amount": "",
        "subtotal": "",
        "tax": "",
        "discount": "",
        "total": "",
        "currency": "",
        "payment_method": "",
        "account_number": "",
        "routing_number": "",
        "bank_name": "",
        "items": [],
        "document_type": "",
        "notes": "",
        "text": f"OCR performed on: {image_path}"
    }

    # Overwrite fields with parsed result if present
    for key in fields:
        if key in parsed:
            fields[key] = parsed[key]

    return fields
