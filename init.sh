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