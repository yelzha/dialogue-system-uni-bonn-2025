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

echo "Finished!!!"