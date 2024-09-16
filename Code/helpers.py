import os
import torch

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.pipeline_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.pipelines.pipeline_loading_utils import _get_pipeline_class, load_sub_model, ALL_IMPORTABLE_CLASSES
from diffusers import pipelines
from prune import prune_mag_outlier, prune_wanda_outlier, check_sparsity, prune_magnitude, prune_wanda

def load_sub_module(input_dict, full_model_pipeline_class, model_path, cuda=True):

    name, (library_name, class_name) = list(input_dict.items())[0]

    config_dict = full_model_pipeline_class.load_config(model_path)

    is_pipeline_module = hasattr(pipelines, library_name)
    importable_classes = ALL_IMPORTABLE_CLASSES
    loaded_sub_model = None

    pipeline_class = _get_pipeline_class(
        full_model_pipeline_class,
        config_dict,
    )

    torch_dtype = torch.float16

    model = load_sub_model(
        library_name=library_name,
        class_name=class_name,
        importable_classes=importable_classes,
        pipelines=pipelines,
        is_pipeline_module=is_pipeline_module,
        pipeline_class=pipeline_class,
        torch_dtype=torch_dtype,
        provider=None,
        sess_options=None,
        device_map=None,
        max_memory=None,
        offload_folder=None,
        offload_state_dict=False,
        model_variants={},
        name=name,
        from_flax=False,
        variant=None,
        low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT,
        cached_folder=model_path,
    )

    if cuda:
        model.to("cuda")

    return model

def prune_and_save_clip_sd2(model_path, pruned_model_path, prune_method, sparsity, verbose=True):

    if prune_method not in ["wanda", "magnitude", "owl_wanda", "owl_magnitude"]:
        print(prune_method)
        raise ValueError("Invalid Prune Method")
    
    text_encoder = load_sub_module({'text_encoder': ['transformers', 'CLIPTextModel']}, StableDiffusionPipeline, model_path)
    tokenizer = load_sub_module({"tokenizer": ["transformers", "CLIPTokenizer"]}, StableDiffusionPipeline, model_path, cuda=False)

    if prune_method == "wanda":
        prune_func = lambda x,y: prune_wanda(x, y, tokenizer)
    elif prune_method == "magnitude":
        prune_func = prune_magnitude
    elif prune_method == "owl_wanda":
        prune_func = lambda x,y: prune_wanda_outlier(x, y, tokenizer, 7, 0.08)
    elif prune_method == "owl_magnitude":
        prune_func = lambda x,y: prune_mag_outlier(x, y, tokenizer, 7, 0.08)

    snr = prune_func(text_encoder, sparsity)

    if verbose:
        print(f"SNR: {snr}")
        print(f"Expected Sparsity: {sparsity}")
        print(f"Real Sparsity: {check_sparsity(text_encoder, verbose=False)}")

    if os.path.isfile(pruned_model_path + "/text_encoder/model.safetensors"):
        os.remove(pruned_model_path + "/text_encoder/model.safetensors")

    text_encoder.save_pretrained(pruned_model_path + "/text_encoder")

def prune_and_save_unet_sd2(model_path, pruned_model_path, prune_method, sparsity, verbose=True):

    if prune_method not in ["magnitude"]:
        print(prune_method)
        raise ValueError("Invalid Prune Method")
    
    if prune_method == "magnitude":
        prune_func = prune_magnitude
    
    unet = load_sub_module({'unet': ['diffusers', 'UNet2DConditionModel']}, StableDiffusionPipeline, model_path)

    snr = prune_func(unet, sparsity)

    if verbose:
        print(f"SNR: {snr}")
        print(f"Expected Sparsity: {sparsity}")
        print(f"Real Sparsity: {check_sparsity(unet, verbose=False)}")

    if os.path.isfile(f"./{pruned_model_path}/unet/diffusion_pytorch_model.safetensors"):
        os.remove(f"./{pruned_model_path}/unet/diffusion_pytorch_model.safetensors")

    unet.save_pretrained(f"{pruned_model_path}/unet")
    