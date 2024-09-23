import argparse
import json
import os
import tqdm
import shutil
import torch
import clip
import re

import numpy as np

from PIL import Image
from Code.helpers import prune_and_save_clip_sd2, prune_and_save_unet_sd2
from diffusers import StableDiffusionPipeline

def norm_images(path):
    norm_path = path+"_norm"
    if os.path.isdir(norm_path):
        print("Normalised dataset already exists")
        return
    
    os.mkdir(norm_path)
    for file_path in os.listdir(path):
        image_path = path + "/" + file_path
        image = Image.open(image_path).convert("RGB").resize((299, 299), resample=Image.BICUBIC)
        image.save(norm_path + "/" + file_path)

def setup_dataset(mscoco_path, seed):
    """
        Setup the MSCOCO dataset.
    """
    captions_train2017_path = mscoco_path + 'annotations/captions_train2017.json'
    unique_captions_train2017_path = mscoco_path + 'annotations/unique_captions_train2017.json'
    train2017_images_path = mscoco_path + "train2017/"

    # Get all captions
    with open(captions_train2017_path) as f:
        train_data = json.load(f)

    train_image_list   = train_data["images"]
    train_caption_list = train_data["annotations"]

    id_to_ind = {}
    for i, image_dict in enumerate(train_image_list):
        id_to_ind[image_dict["id"]] = i

    # Extract all captions for unique images
    unique_train_caption_list = []
    seen_image_ids = set()

    for image_caption in train_caption_list:
        if image_caption["image_id"] not in seen_image_ids:
            seen_image_ids.add(image_caption["image_id"])
            unique_train_caption_list.append(image_caption)

    unique_train_data = {}

    for key, value in train_data.items():
        if key != "annotations":
            unique_train_data[key] = value
        else:
            unique_train_data[key] = unique_train_caption_list

    # Save only one caption for each image
    with open(unique_captions_train2017_path, "w") as outfile:
        json.dump(unique_train_data, outfile)

    # Load saved unique image captions
    with open(unique_captions_train2017_path) as f:
        train_data = json.load(f)

    train_image_list   = train_data["images"]
    train_caption_list = train_data["annotations"]

    id_to_ind = {}
    for i, image_dict in enumerate(train_image_list):
        id_to_ind[image_dict["id"]] = i

    np.random.seed(seed)
    random_inds = np.random.choice(len(train_caption_list), 10000, replace=False)

    # Get subset of images for this seed
    new_data_folder = mscoco_path + f"train2017_{seed}/"

    if not os.path.isdir(new_data_folder):
        os.mkdir(new_data_folder)
        new_data_folder_files = []
    else:
        new_data_folder_files = os.listdir(new_data_folder)

    print(len(random_inds))

    for ind in tqdm(random_inds):
        image_id = train_caption_list[ind]["image_id"]
        image_ind = id_to_ind[image_id]
        image_filename = train_image_list[image_ind]["file_name"]
        # image_files.append(image_filename)
        # if not os.path.isfile(new_data_folder + file_name):
        if image_filename not in new_data_folder_files: 
            shutil.copy(train2017_images_path + image_filename, new_data_folder + image_filename)
    
    norm_images(new_data_folder)

def extract_fid_from_text(path, exp_id):

    with open(path, 'r') as file:
        content = file.read()

    # Apply regex to find the decimal after "FID:"
    pattern = r'FID:\s*(\d+\.\d+)'
    match = re.search(pattern, content)

    if match:
        fid_value = float(match.group(1))
        print(f"The FID is {fid_value}")
    else:
        print("No FID value found in the file.")
        return

    with open("experiments.json", "r") as infile:
        exps = json.load(infile)

    if "FID" not in exps[exp_id].keys():
        exps[exp_id]["FID"] = fid_value

    with open("experiments.json", "w") as outfile:
        json.dump(exps, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='sd-2', help='Model path of unpruned model. If not present will download and save at this path.')
    parser.add_argument('--save_model_path', type=str, default='sd-2', help='Path to save pruned model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for selecting the MSCOCO data for evaluation')
    parser.add_argument('--text_pruning_method', type=str, default='magnitude', help="Pruning algorithm to be applied for text portion of model.")
    parser.add_argument('--image_pruning_method', type=str, default='magnitude', help="Pruning algorithm to be applied for image portion of model.")
    parser.add_argument('--text_sparsity', type=float, default=0, help="Pruning sparsity for text portion of model.")
    parser.add_argument('--image_sparsity', type=float, default=0, help="Pruning sparsity for image portion of model.")
    parser.add_argument('--evaluate', type=bool, default=True, help="Pruning sparsity for image portion of model.")
    parser.add_argument('--metric_images', type=int, default=10000, help="Number of images to be generated for evaluation metrics.")
    parser.add_argument('--save_metric_data_path', type=str, default="sd-2-data/", help="Path to which images generated for metric are saved.")
    parser.add_argument('--mscoco_path', type=str, default="MSCOCO/", help="Path to saved MSCOCO directory with train2017 images and annotation directory.")
    args = parser.parse_args()

    model_path = args.model_path
    pruned_model_path = args.save_model_path

    text_prune_method = args.text_prune_method
    image_prune_method = args.image_prune_method
    
    text_sparsity = float(args.text_sparsity)
    image_sparsity = float(args.image_sparsity)

    if args.image_prune_method != "magnitude":
        raise ValueError("Only magnitude pruning is valid for image portion of model")

    if text_sparsity == 0 and image_sparsity == 0:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.to("cuda")

        model_conf = "base"

    else:
        if text_sparsity != 0:
            prune_and_save_clip_sd2(model_path, pruned_model_path, text_prune_method, text_sparsity)
        print()
        if image_sparsity != 0:
            prune_and_save_unet_sd2(model_path, pruned_model_path, image_prune_method, image_sparsity)

        pipe = StableDiffusionPipeline.from_pretrained(pruned_model_path, torch_dtype=torch.float16)
        pipe.to("cuda")

        model_conf = f"{text_prune_method}-{text_sparsity}__{image_prune_method}-{image_sparsity}"

    if args.evaluate:

        setup_dataset(args.mscoco_path, args.mscoco_seed)

        data_path = args.save_metric_data_path
        new_data_folder = data_path + model_conf
        if not os.path.isdir(new_data_folder):
            os.mkdir(new_data_folder)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        captions_train2017_path = args.mscoco_path + 'annotations/unique_captions_train2017.json'

        with open(captions_train2017_path) as f:
            train_data = json.load(f)

        train_image_list   = train_data["images"]
        train_caption_list = train_data["annotations"]

        id_to_ind = {}
        for i, image_dict in enumerate(train_image_list):
            id_to_ind[image_dict["id"]] = i

        caption_shuffle_seed = int(args.seed)
        np.random.seed(caption_shuffle_seed)
        random_inds = np.random.choice(len(train_caption_list), 10000, replace=False)

        all_scores = []

        for ind in random_inds:
            caption = train_caption_list[ind]["caption"]
            image_id = train_caption_list[ind]["image_id"]
            image_ind = id_to_ind[image_id]
            image_filename = train_image_list[image_ind]["file_name"]

            if os.path.isfile(new_data_folder + "/" + image_filename):
                generated_image = Image.open(new_data_folder + "/" + image_filename)
            else:
                generated_image = pipe(caption).images[0]
                generated_image.save(new_data_folder + "/" + image_filename)
            
            generated_image_tensor = preprocess(generated_image).unsqueeze(0).to(device)
            caption_text = clip.tokenize([caption]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(generated_image_tensor)
                text_features = model.encode_text(caption_text)

            clip_score = torch.nn.functional.cosine_similarity(image_features, text_features).item()
            all_scores.append(clip_score)

        average_clip_score = np.average(all_scores)
        print("CLIPScore:", average_clip_score)

        if os.path.isfile("experiments.json"):
            with open("experiments.json", "r") as infile:
                exps = json.load(infile)
        else:
            exps = {}

        exp_no = None
        for i, model_dict in exps.items():
            if model_dict["config"] == model_conf:
                exp_no = i
        
        if exp_no is not None:
            exp_id = exp_no
            exps[exp_no]["CLIPScore"] = average_clip_score
        else:
            exp_id = len(exps)
            exps[len(exps)] = {
                "dataset_path" : data_path + model_conf,
                "caption_seed" : int(caption_shuffle_seed),
                "text_prune_method" : text_prune_method,
                "image_prune_method" : image_prune_method,
                "text_sparsity" : float(text_sparsity),
                "image_sparsity" : float(image_sparsity),
                # "FID" : fid,
                "config": model_conf,
                "CLIPScore": average_clip_score
            }

        print(exps)

        with open("experiments.json", "w") as outfile:
            json.dump(exps, outfile)
        
        os.system(f"python -m pytorch_fid {args.mscoco}/train2017_{args.seed}_norm/ {new_data_folder} > fid_output.txt")

        extract_fid_from_text("fid_output.txt", exp_id)

if __name__ == "__main__":


    main()