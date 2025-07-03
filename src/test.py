from modules.ocr_parser import parse_image
from modules.rag_store import init_vectorstore, add_doc

vectorstore = init_vectorstore()
doc = parse_image("docs/example_invoice.png")
add_doc(vectorstore, doc)
print("Document added!")
