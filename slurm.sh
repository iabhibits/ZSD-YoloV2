#! /bin/bash
#SBATCH --job-name=ZSD_exp_2   # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --output=LOG/yolo_voc_%j.txt  # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=q2h_48h-1G

NV_GPU=2 nvidia-docker run --device=/dev/nvidia2 --rm -v /home/abhishek/Desktop/YoloV2/:/workspace abhi/pytorch20.01  bash run_yolo_voc.sh
