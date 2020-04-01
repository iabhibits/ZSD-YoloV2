#! /bin/bash
#SBATCH --job-name=ZSD_exp_2   # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --output=LOG/yolo_voc_%j.txt  # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q1m_2h-4G

NV_GPU=2 nvidia-docker run --device=/dev/nvidia2 --rm -v /home/abhishek/Desktop/YoloV2/:/workspace abhi_pytorch  bash yolo_voc_test.sh
