#!/bin/bash
#set a job name
#SBATCH --job-name=freedom
#a file for job output, you can check job progress
#SBATCH --output=logs/run1_%j.out
# a file for errors from the job
#SBATCH --error=logs/run1_%j.err
#time you think you need: default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=4:00:00
#number of cores you are requesting
#SBATCH --cpus-per-task=20
#memory you are requesting
#SBATCH --mem=4Gb
#partition to use
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1

module load anaconda3/2022.05
source activate FREEDOM
module load cuda/12.1

srun $1