# SD2-Pruning
Pruning Implementation for Stable Diffusion 2
Used to prune Stable Diffusion to required sparsities.
Also has evaluation functionality to provide FID and CLIPScore metrics.

# Usage

Install all packages mentioned in requirements.txt.

Download train images and annotations for the 2017 MSCOCO training dataset from https://cocodataset.org/#download

Extract the zip files for images and annotations and place them in one folder.

Below is an example for pruning Stable Diffusion 2 to 50% and getting the FID and CLIP Score.

```sh
python main.py \
    --model_path sd-2 \
    --save_model_path sd-2-pruned \
    --seed 0 \
    --text_pruning_method magnitude \
    --image_pruning_method magnitude \
    --text_sparsity 0.5 \
    --image_sparsity 0.5 \
    --evaluate True \
    --metric_images 10000 \
    --save_metric_data_path sd-2-data/ \
    --mscoco_path MSCOCO/ \
```

The arguments are to be used as follows:
- `--model_path`: Model path of unpruned model. If not present will download and save at this path.
- `--save_model_path`: Path to save pruned model.
- `--seed`: Seed for selecting the MSCOCO data for evaluation
- `--text_pruning_method`: Pruning algorithm to be applied for text portion of model.
- `--image_pruning_method`: Pruning algorithm to be applied for image portion of model.
- `--text_sparsity`: Pruning sparsity for text portion of model.
- `--image_sparsity`: Pruning sparsity for image portion of model.
- `--evaluate`: Pruning sparsity for image portion of model.
- `--metric_images`: Number of images to be generated for evaluation metrics.
- `--save_metric_data_path`: Path to which images generated for metric are saved.
- `--mscoco_path`: Path to saved MSCOCO directory with train2017 images and annotation directory.

FID and CLIP Score are saved in experiments.json.
