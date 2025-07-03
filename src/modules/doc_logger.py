# modules/doc_logger.py

import json
from pathlib import Path

LOG_FILE = Path("data/parsed_docs.jsonl")
VECTORSTORE_FILE = Path("data/vectorstore_docs.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_doc(parsed_data: dict):
    """Append the full parsed document to a .jsonl log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False)
        f.write("\n")

def export_vectorstore_to_jsonl(vectorstore):
    """
    Export all documents from the vector store to a .jsonl file.
    Each line will contain a JSON object with 'text' and 'metadata'.
    """
    docs = vectorstore.get()['documents']
    metadatas = vectorstore.get()['metadatas']

    with open(VECTORSTORE_FILE, "w", encoding="utf-8") as f:
        for text, metadata in zip(docs, metadatas):
            json_line = {
                "text": text,
                "metadata": metadata
            }
            json.dump(json_line, f, ensure_ascii=False)
            f.write("\n")

