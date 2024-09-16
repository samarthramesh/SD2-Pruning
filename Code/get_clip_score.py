import torch
import clip
import sys
import json
import os

import numpy as np

from PIL import Image
from helpers import prune_and_save_clip_sd2, prune_and_save_unet_sd2
from diffusers import StableDiffusionPipeline

def main(text_prune_method, image_prune_method, text_sparsity, image_sparsity, caption_shuffle_seed=0, parallel_key=0):

    if image_prune_method != "magnitude":
        raise ValueError("Only magnitude pruning is valid for image portion of model")

    text_sparsity = float(text_sparsity)
    image_sparsity = float(image_sparsity)

    model_path = "./sd-2"
    pruned_model_path = f"./sd-2-pruned-{parallel_key}"

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

    data_path = "/mnt/parscratch/users/acp23snr/sd-2-data/"
    new_data_folder = data_path + model_conf
    if not os.path.isdir(new_data_folder):
        os.mkdir(new_data_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    captions_train2017_path = '/mnt/parscratch/users/acp23snr/MSCOCO/annotations/unique_captions_train2017.json'

    with open(captions_train2017_path) as f:
        train_data = json.load(f)

    train_image_list   = train_data["images"]
    train_caption_list = train_data["annotations"]

    id_to_ind = {}
    for i, image_dict in enumerate(train_image_list):
        id_to_ind[image_dict["id"]] = i

    caption_shuffle_seed = int(caption_shuffle_seed)
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

    with open("Results/experiments.json", "r") as infile:
        exps = json.load(infile)

    exp_no = None
    for i, model_dict in exps.items():
        if model_dict["config"] == model_conf:
            exp_no = i
    
    if exp_no is not None:
        exps[exp_no]["CLIPScore"] = average_clip_score
    else:
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

    with open("Results/experiments.json", "w") as outfile:
        json.dump(exps, outfile)




if __name__ == "__main__":

    text_prune_method = sys.argv[1]
    image_prune_method = sys.argv[2]
    text_sparsity = sys.argv[3]
    image_sparsity = sys.argv[4]
    caption_shuffle_seed = sys.argv[5]
    parallel_key = sys.argv[6]

    main(text_prune_method, image_prune_method, text_sparsity, image_sparsity, caption_shuffle_seed, parallel_key)