# modules/rag_store.py

import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import CHROMA_DB_DIR

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
    Store a parsed document (OCR result) in the vector store with safe metadata.
    Complex fields like lists are stringified for compatibility.
    """

    # Define all expected fields with defaults
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
        "text": ""
    }

    fields.update(parsed_data)

    # Filter and convert non-scalar values (list/dict) to JSON strings
    def safe_value(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return json.dumps(v, ensure_ascii=False)

    clean_metadata = {k: safe_value(v) for k, v in fields.items()}

    doc = Document(
        page_content=fields["text"] or "No OCR text found.",
        metadata=clean_metadata
    )

    vectorstore.add_documents([doc])
    vectorstore.persist()
