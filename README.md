# Project Roadmap and To-Do List

A local AI-powered financial assistant that reads bills and checks from images, extracts relevant data using vision-language models, and logs structured information into Excel or a database.

---

## Phase 1: Core Functionality

### Image Input and Handling
- [x] Accept image uploads (JPG, PNG, PDF)
- [ ] Support drag-and-drop interface (Gradio or Streamlit)
- [ ] Validate file type and resolution

### Vision-Language Extraction
- [x] Integrate with Donut / LLaVA for OCR + layout parsing
- [ ] Switch to PaddleOCR fallback if needed
- [ ] Auto-detect document type (bill or check)
- [ ] Extract key fields:
  - Vendor / Payee
  - Amount
  - Date
  - Invoice or Check Number

### Data Structuring
- [x] Normalize extracted data to dictionary
- [x] Convert to pandas DataFrame
- [ ] Validate extracted fields and apply cleanup rules

---

## Phase 2: Data Logging

### Excel Integration
- [x] Append structured data to an Excel file
- [x] Handle file not found or empty case
- [ ] Create separate sheets for bills and checks
- [ ] Add timestamp for log entry

### ☐ API/Database Option (Optional)
- [ ] Create SQLite or Postgres schema
- [ ] Replace Excel logging with DB insert
- [ ] Add config toggle: `use_excel = True/False`

---

## Phase 3: Data Analysis

### ☐ Prompt-Driven Insights
- [ ] Integrate local LLM (Mistral / Phi-3 via Ollama)
- [ ] Create prompt template for financial Q&A
- [ ] Build LLMChain to answer queries about:
  - Total spend by vendor
  - Spending trends over time
  - Outstanding unpaid bills

### ☐ Summary Reports
- [ ] Generate summary report as Excel output
- [ ] Export monthly report with charts (optional)

---

## Phase 4: Application UI

### ☐ Gradio or Streamlit Interface
- [ ] Upload or capture image
- [ ] Show extracted fields in preview
- [ ] Allow manual corrections before saving
- [ ] Log result and show success message

---

## Phase 5: Testing and Packaging

### ☐ Unit and Integration Tests
- [ ] Test image parsing accuracy
- [ ] Test Excel writing and merging logic
- [ ] Add test images for regression

### ☐ Packaging and Deployment
- [ ] Create requirements.txt and setup.py
- [ ] Add CLI support: `python agent.py --input bill.jpg`
- [ ] Add Dockerfile for reproducible local setup

---

## Phase 6: Documentation

### ☐ Project Docs
- [x] README.md with overview and usage
- [ ] CONTRIBUTING.md
- [ ] Examples and sample input images
- [ ] Model benchmarks and comparison

---



## Setup Steps
```sh
#!/bin/bash
=========initialization start=========
======================================
mkdir -p datasets
curl -L -o dataset/high-quality-invoice-images-for-ocr.zip https://www.kaggle.com/api/v1/datasets/download/osamahosamabdellatif/high-quality-invoice-images-for-ocr


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

# ~/nlp_lab/run_test.sh:
#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --output=slurm_output.txt   # Log everything here

module load Miniforge3
module load git/2.41.0-GCCcore-12.3.0-nodocs


#conda create -p /home/s06zyelt/nlp_lab/env python=3.10 -y
source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/nlp_lab/env

pip install numpy pandas
pip install openai==0.28.1
pip install sacrebleu
pip install git+https://github.com/openai/human-eval.git

python -c "import numpy, pandas, openai; print('All good')"
python -c "from human_eval.data import read_problems; print('human_eval works')"



export OLLAMA_HOST=127.0.0.1:11500
ollama serve &
sleep 5
ollama run qwen3:0.6b || true

python ollama_test.py

echo "Finished!!!"








# ~/nlp_lab/ollama_test.py:
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
