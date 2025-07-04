import base64
import json
import requests
from PIL import Image
from io import BytesIO
import re
from config import OLLAMA_BASE_URL, OLLAMA_OCR_MODEL

# Instruction prompt for structured invoice/receipt extraction
instruction = (
    "You are given a scanned or photographed image of a receipt, invoice, or check.\n"
    "Your task is to extract structured data from the image and return it in a strict JSON format.\n"
    "\n"
    "### Field Descriptions:\n"
    "- invoice_number: Invoice No., Bill No., Ref No.\n"
    "- check_number: Check No., Cheque ID\n"
    "- po_number: PO No., Purchase Order, Order Ref\n"
    "- vendor: The seller's company or individual name\n"
    "- vendor_address: Full postal address of the vendor\n"
    "- customer_name: The buyer’s name or organization\n"
    "- customer_address: Shipping or billing address of the customer\n"
    "- date: The invoice or receipt issue date\n"
    "- due_date: Payment due date, if present\n"
    "- payment_date: Actual payment date, if available\n"
    "- amount: Total value before tax or discount\n"
    "- subtotal: Pre-tax subtotal (sometimes just labeled as 'Amount')\n"
    "- tax: Total tax amount (VAT, GST, Sales Tax, etc.)\n"
    "- VAT: Value-added tax amount, if specified\n"
    "- discount: Any discount, rebate, or promo; convert percent to absolute if needed\n"
    "- total: Final total payable amount\n"
    "- currency: Currency code (e.g., USD, EUR, GBP, JPY); infer from symbol if needed\n"
    "- payment_method: Credit card, bank transfer, PayPal, cash, etc.\n"
    "- account_number: Bank account or payment account number if present\n"
    "- routing_number: Routing/IBAN/SWIFT number if present\n"
    "- bank_name: Bank name if mentioned\n"
    "- items: List of purchased items or services with quantity, price, and total\n"
    "- document_type: One of: 'invoice', 'receipt', or 'check'\n"
    "- notes: Any extra notes, remarks, or memos\n"
    "\n"
    "### Output Format:\n"
    "Respond ONLY with a valid JSON object in the following format:\n"
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
    "  \"VAT\": string,\n"
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
    "\n"
    "### Important Instructions:\n"
    "- Normalize all monetary values as plain numbers (e.g., 1234.56), without currency symbols.\n"
    "- If any field is missing or unclear, set its value to an empty string.\n"
    "- For tax or discount percentages, calculate final values where possible.\n"
    "- Do not include any explanation, summary, or additional text outside the JSON."
    "- If a summary table shows tax/VAT/subtotal/total values, extract them even if they are not labeled explicitly."
    "- Infer monetary values from tables if they are clearly related to totals (e.g. in summary rows)."
    "- If VAT and tax are shown separately, keep both; if only one is present, copy the same value to both fields."
    "- If currency symbols are present (e.g. '$', '€', '£'), infer the currency (e.g., USD, EUR, GBP)."

)


def clean_response(result):
    # Remove markdown-style triple backticks
    cleaned = result.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # remove ```json
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # remove ```
    return cleaned.strip()


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
            "max_tokens": 4096
        }
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]
        cleaned_result = clean_response(result)
        parsed = json.loads(cleaned_result)

        print(parsed) # to see the parsed image
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
