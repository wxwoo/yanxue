#!/bin/bash

#SBATCH --job-name=chess_game
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=liuzifeng@sjtu.edu.cn

source activate your_conda_env
python [your_code.py]