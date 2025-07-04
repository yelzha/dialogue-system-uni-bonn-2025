# modules/rag_store.py

import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from config import CHROMA_DB_DIR

import os
import time


def init_vectorstore():
    """
    Initialize or load Chroma vector store using a local embedding model.
    This avoids OpenAI and is compatible with fully local RAG setups.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        collection_name="checks",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )


def add_doc(vectorstore, parsed_data):
    """
    Store a parsed document (OCR result) in the vector store with rich page content and metadata.
    This improves semantic retrieval by embedding actual content, not just placeholder text.
    """

    # Define schema with all expected fields
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
        "items": [],  # list of dicts
        "document_type": "",
        "notes": "",
        "text": ""  # raw OCR fallback
    }

    fields.update(parsed_data)

    # ⬇ Build searchable text content for embedding
    page_text = f"""
        Document Type: {fields['document_type']}
        Vendor: {fields['vendor']}
        Vendor Address: {fields['vendor_address']}
        Customer: {fields['customer_name']}
        Customer Address: {fields['customer_address']}
        Invoice #: {fields['invoice_number']}
        Check #: {fields['check_number']}
        PO #: {fields['po_number']}
        Date: {fields['date']}
        Due Date: {fields['due_date']}
        Payment Date: {fields['payment_date']}
        Amount: {fields['amount']}
        Total: {fields['total']}
        Tax: {fields['tax']}
        Payment Method: {fields['payment_method']}
        Bank: {fields['bank_name']}
        Notes: {fields['notes']}
        """

    # Add items if available
    if fields["items"]:
        page_text += "\nLine Items:\n"
        for item in fields["items"]:
            item_line = f"- {item.get('qty', '')} x {item.get('item', '')} @ {item.get('price', '')} = {item.get('total', '')}"
            page_text += item_line + "\n"

    # Safe metadata: convert non-scalar types (list/dict) to JSON strings
    def safe_value(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return json.dumps(v, ensure_ascii=False)

    clean_metadata = {k: safe_value(v) for k, v in fields.items()}

    # Create Document for vector store
    doc = Document(
        page_content=page_text.strip(),
        metadata=clean_metadata
    )

    vectorstore.add_documents([doc])
    vectorstore.persist()


def clear_vectorstore(vectorstore):
    """
    Deletes all documents from the current Chroma vectorstore collection.
    """
    try:
        # Use a valid operator like "$any"
        vectorstore._collection.delete(where={"$any": []})
        vectorstore.persist()
        print("Vectorstore cleared.")
    except Exception as e:
        print(f"Failed to clear vectorstore: {e}")



