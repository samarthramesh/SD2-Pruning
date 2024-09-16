import json
import os
import sys

def main(num_parallel):

    num_parallel = int(num_parallel)

    with open("Experiments/future_experiments.json", "r") as infile:
        future_exps = json.load(infile)

    with open("Results/experiments.json", "r") as infile:
        exps = json.load(infile)

    completed_configs = [completed_exp_dict["config"] for completed_exp_dict in exps.values()]
    completed_exp_ids = [int(e) for e in exps.keys()]

    exp_ids = []

    for i, exp_dict in future_exps.items():
        i = int(i)
        config = f"{exp_dict['text_prune_method']}-{float(exp_dict['text_sparsity'])}__{exp_dict['image_prune_method']}-{float(exp_dict['image_sparsity'])}"
        if config in completed_configs and i != 23:
            continue
        
        used_exp_ids = completed_exp_ids + exp_ids
        if i in used_exp_ids or i != len(used_exp_ids):
            i = len(used_exp_ids)

        parallel_key = i % num_parallel

        with open(f"Experiments/Scripts/exp_{i}.sh", "w") as rsh:
            rsh.write(f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --job-name=exp_{i}  # Replace JOB_NAME with a name you like
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --mem=42G  # Request 5 gigabytes of real memory (mem)
#SBATCH --output=Experiments/Output/exp_{i}.txt  # This is where your output and errors are logged
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=snramesh1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified
#SBATCH --mail-type=FAIL
#SBATCH --time=9:00:00

module load Anaconda3/2022.05
source activate llms

export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

rm -rf sd-2-pruned
cp -r sd-2 sd-2-pruned-{parallel_key}

python Code/get_clip_score.py {exp_dict['text_prune_method']} {exp_dict['image_prune_method']} {exp_dict['text_sparsity']} {exp_dict['image_sparsity']} 0 {parallel_key}

rm -rf sd-2-pruned-{parallel_key}

python Code/norm_images.py /mnt/parscratch/users/acp23snr/sd-2-data/{config}
python -m pytorch_fid /mnt/parscratch/users/acp23snr/MSCOCO/train2017_0_norm/ /mnt/parscratch/users/acp23snr/sd-2-data/{config}_norm/ > Experiments/Output/exp_{i}_fid.txt

rm -rf /mnt/parscratch/users/acp23snr/sd-2-data/{config}_norm/

python Code/extract_fid_from_txt.py {i}
""")
        exp_ids.append(i)

    prev_parallel = [None] * num_parallel
    
    with open("Experiments/main_bash.sh", "w") as rsh:
        rsh.write("#!/bin/bash\n")
        for i, exp_id in enumerate(exp_ids):
            
            parallel_key = exp_id % num_parallel

            if prev_parallel[parallel_key] is None:
                rsh.write(f"jid{exp_id}=$(sbatch Experiments/Scripts/exp_{exp_id}.sh)\n")
                rsh.write(f"jid{exp_id}=$(echo $jid{exp_id} | tr -dc '0-9')\n")
                prev_parallel[parallel_key] = exp_id
            else:
                rsh.write(f"jid{exp_id}=$(sbatch --dependency=afterany:$jid{prev_parallel[parallel_key]} Experiments/Scripts/exp_{exp_id}.sh)\n")
                rsh.write(f"jid{exp_id}=$(echo $jid{exp_id} | tr -dc '0-9')\n")
                prev_parallel[parallel_key] = exp_id

            # if i == 0:
            #     rsh.write(f"jid{exp_id}=$(sbatch Experiments/Scripts/exp_{exp_id}.sh)\n")
            #     rsh.write(f"jid{exp_id}=$(echo $jid{exp_id} | tr -dc '0-9')\n")
            # else:
            #     rsh.write(f"jid{exp_id}=$(sbatch --dependency=afterany:$jid{exp_ids[i-1]} Experiments/Scripts/exp_{exp_id}.sh)\n")
            #     rsh.write(f"jid{exp_id}=$(echo $jid{exp_id} | tr -dc '0-9')\n")


    os.system("source Experiments/main_bash.sh")

# with open("Results/experiments.json", "w") as outfile:
#         json.dump(exps, outfile)

if __name__ == "__main__":
    num_parallel = sys.argv[1]

    main(num_parallel)


