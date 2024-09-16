# remove redundant outlier extraction
# check position_ids usage

import numpy as np
import torch
import torch.nn as nn

from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection
from transformers.models.t5 import T5EncoderModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

def get_prune_layers(model):
    if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
        return model.text_model.encoder.layers, model.text_model.config.max_position_embeddings, model.text_model.config.hidden_size
    elif isinstance(model, T5EncoderModel):
        return model.encoder.block, 512, model.config.d_model
    elif isinstance(model, UNet2DConditionModel):
        return list(model.children()), None, None
    else:
        raise ValueError("Invalid Model Provided for Pruning")

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prune_magnitude(model, sparsity_ratio, prune_n=0, prune_m=0):
    """
    Prune CLIP Text Encoder model by weight magnitude.
    """
    layers, _, _ = get_prune_layers(model)

    signal = 0
    noise = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                if isinstance(sparsity_ratio, float):
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio)-1].cpu()
                    W_mask = (W_metric<=thresh)
                else:
                    thresh1 = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio[0])-1].cpu()
                    thresh2 = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*sparsity_ratio[1])-1].cpu()
                    W_mask = (thresh1<=W_metric) & (W_metric<=thresh2)

            signal += torch.sum(((W[~W_mask].cpu())**2)*0.001).item()
            noise += torch.sum(((W[W_mask].cpu())**2)*0.001).item()

            W[W_mask] = 0

    if noise == 0:
        return 0
    else:
        return 10*np.log10(signal/noise)

def check_sparsity(model, verbose=True):
    """
    Check sparsity of model.
    """

    layers, _, _ = get_prune_layers(model)
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
    
        if verbose:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    return float(count)/total_params

def prepare_calibration_input(model, dataset, tokenizer, device, nsamples):
    layers, max_context_length, hidden_size = get_prune_layers(model)

    dataset_iter = iter(dataset)
    prompts = [next(dataset_iter)["eng_caption"] for _ in range(nsamples)]

    dataloader = []
    for prompt in prompts:
        tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        dataloader.append(tokens.input_ids)

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((128, model.text_model.config.max_position_embeddings, model.text_model.config.hidden_size), dtype=dtype, device=device)
    inps = torch.zeros((128, max_context_length, hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):

            if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):

                inps[cache['i']] = inp
                cache['i'] += 1
                cache["attention_mask"] = args[0]
                cache["causal_attention_mask"] = args[1]
                cache['position_ids'] = kwargs.get('position_ids')
                raise ValueError
            elif isinstance(model, T5EncoderModel):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache["attention_mask"] = kwargs.get('attention_mask')
                raise ValueError
            # inps[cache['i']] = inp
            # cache['i'] += 1
            # cache["attention_mask"] = kwargs.get('attention_mask')
                
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            print(model(batch[0].unsqueeze(0).to(device)))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        causal_attention_mask = cache['causal_attention_mask']
        position_ids = cache['position_ids']

        return inps, outs, attention_mask, causal_attention_mask
    
    elif isinstance(model, T5EncoderModel):
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']

        return inps, outs, attention_mask

def prune_wanda(model, sparsity_ratio, tokenizer, hf_dataset_tag='visheratin/laion-coco-nllb', device=torch.device("cuda:0"), nsamples=128):
    """
    Prune CLIP Text Model useing Wanda.
    """
    
    from datasets import load_dataset
    dataset = load_dataset(hf_dataset_tag, split='train', streaming=True)

    with torch.no_grad():
        if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
            inps, outs, attention_mask, causal_attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        elif isinstance(model, T5EncoderModel):
            inps, outs, attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        else:
            raise ValueError("Invalid Model Provided for Pruning")
    signal = 0
    noise = 0

    layers, _, _ = get_prune_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            if isinstance(sparsity_ratio, float):
                indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            else:
                indices = sort_res[1][:,int(W_metric.shape[1]*sparsity_ratio[0]):int(W_metric.shape[1]*sparsity_ratio[1])]

            W_mask.scatter_(1, indices, True)

            signal += torch.sum(((subset[name].weight.data[~W_mask].cpu())**2)*0.001).item()
            noise += torch.sum(((subset[name].weight.data[W_mask].cpu())**2)*0.001).item()

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    # print("Signal", signal)
    # print("Noise", noise)
    if noise == 0:
        return 0
    else:
        return 10*np.log10(signal/noise)

def check_outlier_mean(mask,threshold):

    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio=float(count)/total_params*100    
    return outlier_ratio

def get_outlier_dist(model, sparsity_ratio, tokenizer, hyper_m, lamda, hf_dataset_tag='visheratin/laion-coco-nllb', device=torch.device("cuda:0"), nsamples=128):
    all_layer_ratio=[]
    from datasets import load_dataset
    dataset = load_dataset(hf_dataset_tag, split='train', streaming=True)

    with torch.no_grad():
        if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
            inps, outs, attention_mask, causal_attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        elif isinstance(model, T5EncoderModel):
            inps, outs, attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        else:
            raise ValueError("Invalid Model Provided for Pruning")
        
    # print("inps",inps)
    layers, _, _ = get_prune_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:

            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)

        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            # print ("layer outlier ratio",out_ratio,out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)
    
    # print ("before adjustment",all_layer_ratio)
    
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio_scaled = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * lamda*2))
    all_layer_ratio_scaled=all_layer_ratio_scaled-np.mean(all_layer_ratio_scaled)+(1-sparsity_ratio)

    return all_layer_ratio, all_layer_ratio_scaled

def prune_mag_outlier(model, sparsity_ratio, tokenizer, hyper_m, lamda, hf_dataset_tag='visheratin/laion-coco-nllb', device=torch.device("cuda:0"), nsamples=128, prune_n=0, prune_m=0):

    all_layer_ratio=[]
    from datasets import load_dataset
    dataset = load_dataset(hf_dataset_tag, split='train', streaming=True)

    with torch.no_grad():
        if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
            inps, outs, attention_mask, causal_attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        elif isinstance(model, T5EncoderModel):
            inps, outs, attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        else:
            raise ValueError("Invalid Model Provided for Pruning")
        
    # print("inps",inps)
    layers, _, _ = get_prune_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        layer_wmetric=[]
        for name in subset:

            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)

        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])

        for out_ratio in [hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            # print ("layer outlier ratio",out_ratio,out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)
    
    # print ("before adjustment",all_layer_ratio)
    
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-sparsity_ratio)
    
    # print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
    # print ("after adjustment",all_layer_ratio)
    # print (layers)

    signal = 0
    noise = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                # print(name, W.shape, W_metric.shape, layer_sparsity_ratio)
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            signal += torch.sum(((W[~W_mask].cpu())**2)*0.001).item()
            noise += torch.sum(((W[W_mask].cpu())**2)*0.001).item()

            W[W_mask] = 0

    if noise == 0:
        return 0
    else:
        return 10*np.log10(signal/noise)

def prune_wanda_outlier(model, sparsity_ratio, tokenizer, hyper_m, lamda, hf_dataset_tag='visheratin/laion-coco-nllb', device=torch.device("cuda:0"), nsamples=128):
    ##### calucalte outlier ratio
    
    all_layer_ratio=[]
    from datasets import load_dataset
    dataset = load_dataset(hf_dataset_tag, split='train', streaming=True)

    with torch.no_grad():
        if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
            inps, outs, attention_mask, causal_attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        elif isinstance(model, T5EncoderModel):
            inps, outs, attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        else:
            raise ValueError("Invalid Model Provided for Pruning")
        
    # print("inps",inps)
    layers, _, _ = get_prune_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]
        for name in subset:

            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)    
                
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            # print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        

    # print ("before adjustment",all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-sparsity_ratio)

    # print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
    # print ("after adjustment",all_layer_ratio  )
    
    torch.cuda.empty_cache()
    ############## prune
    
    dataset = load_dataset(hf_dataset_tag, split='train', streaming=True)

    with torch.no_grad():
        if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
            inps, outs, attention_mask, causal_attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        elif isinstance(model, T5EncoderModel):
            inps, outs, attention_mask = prepare_calibration_input(model, dataset, tokenizer, device, nsamples)
        else:
            raise ValueError("Invalid Model Provided for Pruning")

    signal = 0
    noise = 0

    layers, _, _ = get_prune_layers(model)
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()


        for name in subset:

            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio= 1-all_layer_ratio[i]

            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            signal += torch.sum(((subset[name].weight.data[~W_mask].cpu())**2)*0.001).item()
            noise += torch.sum(((subset[name].weight.data[W_mask].cpu())**2)*0.001).item()

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(nsamples):
            with torch.no_grad():
                if isinstance(model, CLIPTextModel) or isinstance(model, CLIPTextModelWithProjection):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0]
                elif isinstance(model, T5EncoderModel):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    torch.cuda.empty_cache()
    
    if noise == 0:
        return 0
    else:
        return 10*np.log10(signal/noise)