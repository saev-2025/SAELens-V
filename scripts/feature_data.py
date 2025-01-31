import os
import random
import argparse
import numpy as np
import torch
from PIL import Image
import tqdm
from datasets import load_dataset 
from sae_lens.activation_visualization import (
    load_llava_model,
    load_sae,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--output_path', type=str, default='count_table.txt')
    parser.add_argument('--model_name', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sae_path', type=str, required=True)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_type', type=str, choices=['image', 'text'],
                        default='image')
    args = parser.parse_args()
    return args

def prepare_batch_image_input(processor, device, file_paths, example_prompt):
    images = []
    prompts = []
    for file_path in file_paths:
        image = Image.open(file_path)
        image = image.resize((336, 336))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_prompt + "<image>"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        prompts.append(prompt)
        images.append(image)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors='pt',
        padding=True,
    ).to(device)
    return inputs

def prepare_batch_text_input(processor, device, texts):
    inputs = processor(
        text=texts,
        return_tensors='pt',
        padding=True,
    ).to(device)
    return inputs

def run_model(inputs, hook_language_model, sae, sae_device, is_text=False):
    image_indices=None
    with torch.no_grad():
        if is_text:
            cache = hook_language_model.run_with_cache(
                input=inputs,
                model_inputs=inputs,
                vision=False,
                prepend_bos=True,
                names_filter=lambda name: name == sae.cfg.hook_name,
                return_type="generate",
            )
        else:
            _,image_indices, cache = hook_language_model.run_with_cache(
                input=inputs,
                model_inputs=inputs,
                vision=True,
                prepend_bos=True,
                names_filter=lambda name: name == sae.cfg.hook_name,
                return_type="generate_with_saev",
            )
        tmp_cache = cache[sae.cfg.hook_name]
        tmp_cache = tmp_cache.to(sae_device)
        feature_acts = sae.encode(tmp_cache)
        sae_out = sae.decode(feature_acts)
        del cache
    return image_indices,feature_acts

def save_results(output_path, feature_acts_count_table):
    with open(output_path, "w") as f:
        for index, count in enumerate(feature_acts_count_table):
            f.write(f"{index}: {count}\n")
    print(f"table saves to {output_path}")

def process_images(sampled_png_files, image_path, processor, device, hook_language_model, sae, sae_device, example_prompt, batch_size=32):
    num_files = len(sampled_png_files)
    feature_acts_count_table = np.zeros(65536, dtype=int)

    with tqdm.tqdm(total=num_files) as pbar:
        for i in range(0, num_files, batch_size):
            batch_files = sampled_png_files[i:i+batch_size]
            batch_file_paths = [os.path.join(image_path, f) for f in batch_files]
     
            inputs = prepare_batch_image_input(processor, device, batch_file_paths, example_prompt)
           
            image_indices, feature_act = run_model(inputs, hook_language_model, sae, sae_device)
            feature_act = feature_act.cpu().detach().numpy()
            
            last_token_feature_act = feature_act[:, -1, :]  
            indices = np.where(last_token_feature_act > 1)
            feature_indices = indices[1]
            unique_features, counts = np.unique(feature_indices, return_counts=True)
            feature_acts_count_table[unique_features] += counts
            pbar.update(len(batch_files))
    return feature_acts_count_table

def process_text(dataset, processor, device, hook_language_model, sae,
                 sae_device, text_field, batch_size=32):
    num_samples = len(dataset)
    feature_acts_count_table = np.zeros(65536, dtype=int)

    with tqdm.tqdm(total=num_samples) as pbar:
        for i in range(0, num_samples, batch_size):
            batch = dataset[i:i+batch_size]
            texts = batch[text_field]
            inputs = prepare_batch_text_input(processor, device, texts)
            feature_act = run_model(inputs, hook_language_model, sae,
                                    sae_device, is_text=True)
            feature_act = feature_act.cpu().detach().numpy()
            last_token_feature_act = feature_act[:, -1, :]
            indices = np.where(last_token_feature_act > 1)
            feature_indices = indices[1]
            unique_features, counts = np.unique(feature_indices,
                                                return_counts=True)
            feature_acts_count_table[unique_features] += counts
            pbar.update(len(texts))
    return feature_acts_count_table

def main():
    args = parse_arguments()
    
    random_seed =42
    random.seed(random_seed)

    device = args.device
    sae_device = args.device  

    model_name = args.model_name
    model_path = args.model_path
    sae_path = args.sae_path
    output_path = args.output_path
    sample_size = args.sample_size
    example_prompt="What is shown in the image?"

    processor, vision_model, vision_tower, multi_modal_projector, hook_language_model = load_llava_model(
        model_name, model_path, device
    )
    sae = load_sae(sae_path, sae_device)

    if args.data_type == 'image':
        image_path = ""
        files = os.listdir(image_path)
        png_files = [f for f in files if f.lower().endswith('.png')]
        png_files.sort()
        if sample_size > len(png_files):
            sample_size = len(png_files)
        sampled_png_files = random.sample(png_files, sample_size)
        feature_acts_count_table = process_images(
            sampled_png_files, image_path, processor, device,
            hook_language_model, sae, sae_device, example_prompt)
    elif args.data_type == 'text':
        if args.dataset_name is None:
            raise ValueError("When data_type is text, dataset_name must be provided.")
        dataset = load_dataset(args.dataset_name, split=args.split)
        if sample_size > len(dataset):
            sample_size = len(dataset)
        dataset = dataset.shuffle(seed=random_seed)
        dataset = dataset.select(range(sample_size))
        feature_acts_count_table = process_text(
            dataset, processor, device, hook_language_model, sae,
            sae_device, args.text_field)
    else:
        raise ValueError("Unknown data type:{}".format(args.data_type))

    save_results(output_path, feature_acts_count_table)

if __name__ == "__main__":
    main()
