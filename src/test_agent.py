# test_agent.py

from modules.ocr_parser import parse_image
from modules.rag_store import init_vectorstore, add_doc
from modules.doc_logger import log_doc

# Fake image path (use an uploaded test image in `docs/`)
image_path = "docs/invoice_example_1.png"
parsed = parse_image(image_path)

print("Parsed OCR Output:")
print(parsed)

# Store in vector DB (no LLM, just storage)
vs = init_vectorstore()
add_doc(vs, parsed)

print("\nDocument successfully added to vectorstore.")

# After vectorstore write
log_doc(parsed)
print("Saved to parsed_docs.jsonl ✅")