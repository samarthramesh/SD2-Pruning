import re
import json
import sys

def main(exp_id):

    with open(f'Experiments/Output/exp_{exp_id}_fid.txt', 'r') as file:
        content = file.read()

    # Apply regex to find the decimal after "FID:"
    pattern = r'FID:\s*(\d+\.\d+)'
    match = re.search(pattern, content)

    if match:
        fid_value = float(match.group(1))
        print(f"The FID for exp {exp_id}: is {fid_value}")
    else:
        print("No FID value found in the file.")
        return

    with open("Results/experiments.json", "r") as infile:
        exps = json.load(infile)

    if "FID" not in exps[exp_id].keys():
        exps[exp_id]["FID"] = fid_value

    with open("Results/experiments.json", "w") as outfile:
        json.dump(exps, outfile)

if __name__ == "__main__":
    exp_id = sys.argv[1]

    main(exp_id)

# for exp_id in exps.keys():
#     # Open and read the file
#     with open(f'../Experiments/Output/exp_{exp_id}_fid.txt', 'r') as file:
#         content = file.read()

#     # Apply regex to find the decimal after "FID:"
#     pattern = r'FID:\s*(\d+\.\d+)'
#     match = re.search(pattern, content)

#     if match:
#         fid_value = float(match.group(1))
#         print(f"The decimal value after FID: is {fid_value}")
#     else:
#         print("No FID value found in the file.")

#     if "FID" not in exps[exp_id].keys():
#         exps[exp_id]["FID"] = fid_value