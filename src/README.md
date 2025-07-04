# No More Manual Invoices! (OCR + RAG + LLM via Ollama)
# Team #botz

A local-first AI agent system to parse, store, search, and analyze invoices and checks using OCR, vector databases, and local large language models via Ollama.

### Requirements

- Python 3.9+
- Ollama installed locally: https://ollama.com
- Models pulled via Ollama:
  - ollama pull qwen:vl
  - ollama pull llama3

### Setup Instructions

1. Clone repository:

   git clone https://github.com/yelzha/dialogue-system-uni-bonn-2025.git  
   cd dialogue-system-uni-bonn-2025/src

2. Install dependencies:

   pip install -r requirements.txt

3. Start Ollama:

   export OLLAMA_HOST=127.0.0.1:11501

   ollama serve &

   export OLLAMA_HOST=127.0.0.1:11501
   
   ollama run llama3 || true

   export OLLAMA_HOST=127.0.0.1:11501

   ollama run qwen2.5vl-3b || true

4. Run the Streamlit app:

   streamlit run app.py

## Folder Structure

check-ai-agent/  
├── app.py  
├── config.py  
├── requirements.txt  
├── .gitignore  
├── data/  
├── docs/  
├── modules/   
│   ├── agent_tools.py  
│   ├── analytics.py 
│   ├── doc_logger.py    
│   ├── llm_agent.py   
│   ├── llm_provider.py 
│   ├── ocr_parser.py  
│   └── rag_store.py 
└── README.md

## Usage

1. Upload invoice/check image via UI
2. OCR using Qwen2.5-VL (Ollama)
3. Store parsed data in Chroma vector store
4. Ask questions using RAG + Ollama LLM
5. View analytics

## OCR Output Fields

- invoice_number, check_number, po_number  
- vendor, vendor_address, customer_name, customer_address  
- date, due_date, payment_date  
- amount, subtotal, tax, discount, total, currency  
- payment_method, account_number, routing_number, bank_name  
- document_type, notes, text  
- items[] (item, qty, price, total)

## Stack

- OCR: Qwen2.5-VL via Ollama  
- LLM: LLaMA 3 / Qwen via Ollama  
- Embedding: sentence-transformers/all-MiniLM-L6-v2  
- Vector DB: Chroma  
- Frontend: Streamlit  
- Analytics: Pandas