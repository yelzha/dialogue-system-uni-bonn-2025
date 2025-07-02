#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1


#cd $SLURM_SUBMIT_DIR


module load Miniforge3
module load CUDA/11.7.0
module load git/2.41.0-GCCcore-12.3.0-nodocs

# conda create -p /home/s06zyelt/dialogue-system-uni-bonn-2025/env python=3.10 -y
source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env

# pip install -r requirements.txt
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Run the evaluation script
python src/evaluate_cord_qwen.py

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
