#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=07:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=20G


#cd $SLURM_SUBMIT_DIR


module load Miniforge3
module load CUDA/11.8.0
module load git/2.41.0-GCCcore-12.3.0-nodocs

source /software/easybuild-INTEL_A40/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate /home/s06zyelt/dialogue-system-uni-bonn-2025/env

pip install -r requirements.txt

# Run the evaluation script
echo "++++++++++++++++++++++START+++++++++++++++++++++++++++++"
python -u test/evaluate_cord_qwen.py
echo "++++++++++++++++++++++FINISHED++++++++++++++++++++++++++"
