#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --job-name=acp23snr_dissertation  # Replace JOB_NAME with a name you like
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=82G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=Experiments/Output/exp_1.txt  # This is where your output and errors are logged
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=snramesh1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified
#SBATCH --mail-type=FAIL
#SBATCH --time=24:00:00

module load Anaconda3/2022.05
source activate llms

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

rm -rf sd-2-pruned
cp -r sd-2 sd-2-pruned-1

python Code/get_clip_score.py magnitude magnitude 0.53 0.07 0 1

rm -rf sd-2-pruned-1
