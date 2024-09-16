#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --job-name=acp23snr_dissertation  # Replace JOB_NAME with a name you like
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=82G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=Experiments/Output/exp_1_fid.txt  # This is where your output and errors are logged
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=snramesh1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified
#SBATCH --mail-type=FAIL
#SBATCH --time=1:00:00

module load Anaconda3/2022.05
source activate llms

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

python Code/norm_images.py /mnt/parscratch/users/acp23snr/sd-2-data/magnitude-0.53__magnitude-0.07
python -m pytorch_fid /mnt/parscratch/users/acp23snr/MSCOCO/train2017_0_norm/ /mnt/parscratch/users/acp23snr/sd-2-data/magnitude-0.53__magnitude-0.07_norm/

rm -rf /mnt/parscratch/users/acp23snr/sd-2-data/magnitude-0.53__magnitude-0.07_norm/
