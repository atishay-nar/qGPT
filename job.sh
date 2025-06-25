#!/bin/bash
# This script is used to run the training and sampling processes for a machine learning model.
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gpus=a100:1
#SBATCH --mem=64G
#SBATCH --time=1-0 
#SBATCH -p general
#SBATCH -q public
#SBATCH -o=job.out
#SBATCH -e=job.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anaray84@asu.edu
module load mamba/latest
source activate myenv
python src/train.py