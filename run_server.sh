#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1


#cd $SLURM_SUBMIT_DIR
export OLLAMA_HOST=127.0.0.1:11434

ollama serve &
sleep 5

ollama pull llama3
ollama run llama3 || true

ollama pull qwen2.5vl:3b
ollama run qwen2.5vl:3b || true


module load Miniforge3
module load CUDA/11.8.0
module load git/2.41.0-GCCcore-12.3.0-nodocs

source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env


cd src

pip install -r requirements.txt

# Run the evaluation script
echo "++++++++++++++++++++++START+++++++++++++++++++++++++++++"
streamlit run app.py
echo "++++++++++++++++++++++FINISHED++++++++++++++++++++++++++"




