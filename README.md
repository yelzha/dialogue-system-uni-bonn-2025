# No More Manual Invoices! (OCR + RAG + LLM via Ollama)
# Team #botz

# Demo: https://youtu.be/qRcjYFw-9MU
<img width="500" height="500" alt="adobe-express-qr-code (3)" src="https://github.com/user-attachments/assets/238befe8-1442-4985-b450-ed0202451c0c" />


A local-first AI agent system to parse, store, search, and analyze invoices and checks using OCR, vector databases, and local large language models via Ollama.

---

## TO-DO

### Phase 1: Core Functionality

#### Image Upload and Parsing
- [x] Modular folder structure for OCR, RAG, LLM, analytics, UI
- [x] Accept image uploads (JPG, PNG)
- [x] Parse invoice/check images using Qwen2.5-VL via Ollama

#### Data Extraction
- [x] Extract structured fields (vendor, amount, date, etc.)
- [x] Normalize extracted OCR data into consistent schema

---

### Phase 2: Storage and Retrieval (RAG)

#### Embedding and Storage
- [x] Store documents in Chroma vector store using local sentence-transformer embeddings
- [x] Index structured OCR output using vector search

#### Retrieval and LLM Integration
- [x] Enable Retrieval-Augmented Generation (RAG) with LangChain
- [x] Use LLaMA/Qwen/Mistral via Ollama for question answering
- [x] Possibility to call Tools

---

### Phase 3: Analytics and Reporting

#### Data Aggregation
- [x] Analyze data using Pandas (monthly summaries, top vendors)
- [x] Extract line item totals and vendor grouping

#### Export and Filtering
- [x] Metadata filtering by vendor/date/category (optional)
- [x] Export data to CSV/Excel (optional)

---

### Phase 4: Application UI

#### Frontend
- [x] Streamlit UI for upload, Q&A, preview, and reporting
- [x] Display parsed metadata and items in readable layout

---

### Phase 5: Deployment (optional)

#### Deployment
- [x] Bender Server deployment
- [x] CLI interface for headless execution (optional)

---

## Installation

### Requirements

- Python 3.9+
- Ollama installed locally: https://ollama.com
- Models pulled via Ollama:
  - ollama pull qwen:vl
  - ollama pull llama3

### Setup Instructions

1. Clone repository:
```
   git clone https://github.com/yelzha/dialogue-system-uni-bonn-2025.git  
   cd dialogue-system-uni-bonn-2025/src
```
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Start Ollama:
```
   export OLLAMA_HOST=127.0.0.1:11501
   ollama serve &
```
```
   export OLLAMA_HOST=127.0.0.1:11501
   ollama run llama3 || true
```
```
   export OLLAMA_HOST=127.0.0.1:11501
   ollama run qwen2.5vl:7b || true
```
4. Run the Streamlit app:
```
   cd dialogue-system-uni-bonn-2025
   module load Miniforge3
   conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env
   cd src
   streamlit run app.py
```
6. On your Windows / Linux to publish it for your local computer
```
   ssh -L 8501:localhost:8501 your-username@host-address.de
```
## Folder Structure

dialogue-system-uni-bonn-2025  
├── dataset/  
├── notebooks/  
├── src/  
│   ├── data/  
│   ├── docs/  
│   ├── modules/  
│   │   ├── agent_tools.py  
│   │   ├── analytics.py  
│   │   ├── doc_logger.py  
│   │   ├── llm_agent.py  
│   │   ├── llm_provider.py  
│   │   ├── ocr_parser.py  
│   │   └── rag_store.py  
│   ├── .gitignore  
│   ├── app.py  
│   ├── config.py  
│   ├── README.md  
│   └── requirements.txt  
├── test/  
├── init.sh  
├── ollama_test.py  
├── README.md  
├── requirements.txt  
├── requirements_notebooks.txt  
├── run_code.sh  
├── run_eval_cord_qwen.sh  
├── run_server.sh  
└── run_test.sh  



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



## Setup Steps
```sh
# ~/dialogue-system-uni-bonn-2025/init.sh:
#!/bin/bash
=========initialization start=========
======================================
mkdir -p ~/ollama/bin
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz -C ~/ollama
echo 'export PATH="$HOME/ollama/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
ollama --version



# project part
cd dialogue-system-uni-bonn-2025
mkdir -p datasets
curl -L -o datasets/high-quality-invoice-images-for-ocr.zip https://www.kaggle.com/api/v1/datasets/download/osamahosamabdellatif/high-quality-invoice-images-for-ocr


# double check this place, maybe some mistakes / errors
module load Miniforge3
module load git/2.41.0-GCCcore-12.3.0-nodocs
conda create -p /home/s06zyelt/dialogue-system-uni-bonn-2025/env python=3.10 -y
source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env

cd dialogue-system-uni-bonn-2025
sbatch run_test.sh

==========initialization end==========
======================================
```






```sh
#!/bin/bash
==========code test start=============
======================================

# ~/dialogue-system-uni-bonn-2025/run_test.sh:
#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --output=slurm_output.txt   # Log everything here

module load Miniforge3
module load git/2.41.0-GCCcore-12.3.0-nodocs


#conda create -p /home/s06zyelt/dialogue-system-uni-bonn-2025/env python=3.10 -y
source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env

pip install numpy pandas
pip install openai==0.28.1
pip install sacrebleu
pip install git+https://github.com/openai/human-eval.git

python -c "import numpy, pandas, openai; print('All good')"
python -c "from human_eval.data import read_problems; print('human_eval works')"



export OLLAMA_HOST=127.0.0.1:11434
ollama serve &
sleep 5
ollama run qwen3:0.6b || true

python ollama_test.py

echo "Finished!!!"








# ~/dialogue-system-uni-bonn-2025/ollama_test.py:
import requests

# old port: 11434

response = requests.post(
    'http://localhost:11500/api/generate',
    json={
        'model': 'qwen3:0.6b',
        'prompt': 'What is the capital of France?',
        'stream': False
    }
)

result = response.json()['response']

# Print to console (optional)
print(result)

# Save to a text file
with open('output.txt', 'w') as f:
    f.write(result)

==========code test end===============
======================================
```
