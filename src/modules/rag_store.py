# modules/rag_store.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    Store a parsed document (OCR result) in the vector store with full metadata.
    Includes all known invoice/check fields, even if values are missing.
    """
    # Complete schema
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
        "text": ""
    }

    # Merge with parsed values
    fields.update(parsed_data)

    # Store document
    doc = Document(
        page_content=fields["text"] or "No OCR text found.",
        metadata=fields
    )

    vectorstore.add_documents([doc])
    vectorstore.persist()
