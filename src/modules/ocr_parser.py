import base64
import json
import requests
from PIL import Image
from io import BytesIO
from config import OLLAMA_BASE_URL, OLLAMA_OCR_MODEL

# Instruction prompt, adapted from original
instruction = (
    "Extract and return all structured fields from this receipt or invoice image in the following JSON format:\n"
    "{\n"
    "  \"invoice_number\": string,                 # Also known as Invoice No., Bill No., Ref No.\n"
    "  \"check_number\": string,                   # Can appear as Check No., Cheque ID\n"
    "  \"po_number\": string,                      # May appear as PO No., Purchase Order, Order Ref\n"
    "  \"vendor\": string,                         # The seller’s company name or individual name\n"
    "  \"vendor_address\": string,                 # The full address of the vendor\n"
    "  \"customer_name\": string,                  # The buyer’s name or organization\n"
    "  \"customer_address\": string,               # Address of the buyer or shipping address\n"
    "  \"date\": string,                           # Invoice date; can appear as Issue Date, Billing Date\n"
    "  \"due_date\": string,                       # Payment deadline; may appear as Due On, Payable By\n"
    "  \"payment_date\": string,                   # Actual payment date if present\n"
    "  \"amount\": string,                         # The amount before tax and discount\n"
    "  \"subtotal\": string,                       # The intermediate total (before tax/discount); sometimes labeled as 'Amount'\n"
    "  \"tax\": string,                            # Total tax applied. May be called VAT, GST, Sales Tax. Sometimes shown as percentage (e.g. 10%) — in that case, convert to final tax amount if possible\n"
    "  \"discount\": string,                       # Discount amount, possibly labeled as Rebate, Promo, or shown as percentage — convert to final amount if possible\n"
    "  \"total\": string,                          # Final payable amount. Can be called Total Due, Grand Total, Amount Payable\n"
    "  \"currency\": string,                       # Currency used (e.g. USD, EUR, GBP, ¥, etc.) — infer from symbols if not explicitly stated\n"
    "  \"payment_method\": string,                 # e.g. Credit Card, Wire Transfer, PayPal, Bank Transfer, Cash\n"
    "  \"account_number\": string,                 # If shown, extract the bank account or payment account number\n"
    "  \"routing_number\": string,                 # Bank routing/IBAN/SWIFT if available\n"
    "  \"bank_name\": string,                      # Bank institution name, if mentioned\n"
    "  \"items\": [                                # Line items: list of purchased products or services\n"
    "    {\"item\": string, \"qty\": string, \"price\": string, \"total\": string}\n"
    "  ],\n"
    "  \"document_type\": string,                  # Either 'invoice', 'receipt', or 'check' — infer based on content\n"
    "  \"notes\": string                           # Any additional remarks, memos, or footnotes\n"
    "}\n"
    "\n"
    "Important:\n"
    "- Normalize all monetary values as plain numbers, without currency symbols.\n"
    "- If multiple tax or discount fields are present, sum them into one value where applicable.\n"
    "- If any field is not present or unclear, set it to an empty string.\n"
    "- Your response must contain only the final JSON — no extra explanation or text."
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
        "model": OLLAMA_OCR_MODEL,
        "prompt": instruction,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 8192  # You can increase to 4096 if needed
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
