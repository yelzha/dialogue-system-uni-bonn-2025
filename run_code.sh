#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=00:05:00
#SBATCH --gpus=1
#SBATCH --ntasks=1


#cd $SLURM_SUBMIT_DIR
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_HOST=127.0.0.1:11434

ollama serve &
sleep 5

ollama run qwen3:4b || true

module load Miniforge3
module load git/2.41.0-GCCcore-12.3.0-nodocs

#conda create -p /home/s06zyelt/dialogue-system-uni-bonn-2025/env python=3.10 -y
source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env

pip install -r requirements.txt

# Run the evaluation script
python src/evaluate_cord_qwen.py

cd AgentForest/script
sh run_experiments.sh
# python ollama_test.py

pkill ollama

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
