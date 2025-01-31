from transformers import AutoTokenizer, PreTrainedTokenizerBase,LlavaNextProcessor

import io
import os
import json
from abc import ABC, abstractmethod
import copy
from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+
from tqdm import tqdm

import requests
import librosa
from PIL import Image
from torchvision.io import read_video
import uuid


import torch
import torch.multiprocessing as mp
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from datasets import load_dataset
import argparse

ALLOWED_ATTRIBUTES = ['split_token']
DEFAULT_SPLIT_TOKEN = 'ASSISTANT:'
IGNORE_INDEX = -100


def load_image(image_path: str):
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error occured when dealing with {image_path}")
        raise Exception

def insert_img_token(text, image):
    # do the same for worse
    if isinstance(image, str):
        decoded_images = [load_image(image)]
        num_images = 1
    elif isinstance(image, list):
        decoded_images = [load_image(img) for img in image]
        num_images = len(image)
    elif isinstance(image, Image.Image):
        decoded_images = [image]
        num_images = 1
    else:
        num_images = 0
        decoded_images = None
    
    # processed_text = f"{'<image>' * num_images}{text}"
    processed_text = text
    return processed_text, decoded_images

def safe_add(list1, list2):
    if list1 is None and list2 is None:
        return []
    elif list1 is None:
        return list2.copy()
    elif list2 is None:
        return list1.copy()
    else:
        return list1 + list2
    
def format_sample_cham(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """ Formating input sample, and change the related keys according to the training dataset."""
    """If you are using a different dataset, you need to customize this function or write a new function like the ones below."""
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    input_text = raw_sample['input_text']
    input_img = raw_sample['input_image']
    
    better_text = raw_sample['better_text']
    better_img = raw_sample['better_img']
    
    worse_text = raw_sample['worse_text']
    worse_img = raw_sample['worse_img']
    
    input_text_processed, input_images = insert_img_token(input_text, input_img)
    
    better_text_processed, better_images = insert_img_token(better_text, better_img)
    
    worse_text_processed, worse_images = insert_img_token(worse_text, worse_img)
    
    
    better_text_full = (
        f'{system_prompt}'
        f'{user_prompt.format(input=input_text_processed)}'
        f"{assistant_prompt.format(output=better_text_processed)}"
    )
    
    better_images_full = safe_add(input_images, better_images)

    worse_text_full = (
        f'{system_prompt}'
        f'{user_prompt.format(input=input_text_processed)}'
        f"{assistant_prompt.format(output=worse_text_processed)}"
    )
    
    worse_images_full = safe_add(input_images, worse_images)
    
    return {
        'better_text': better_text_full,
        'worse_text': worse_text_full,
        'better_images': better_images_full,
        'worse_images': worse_images_full,
    }

def format_sample_align_anything_ti2t(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Specified for align-anything ti2t dataset. You can customize a similar function for you own dataset."""
    input_text = f"{raw_sample['question']} <image>"

    input_image = raw_sample['image']
    processed_input_image = [load_image(input_image)]
    if raw_sample['overall_response'] == 1:
        better_text = raw_sample['response_1']
        worse_text = raw_sample['response_2']
    else:
        better_text = raw_sample['response_2']
        worse_text = raw_sample['response_1']
    
    better_text_full = f"{input_text} {better_text}"
    worse_text_full = f"{input_text} {worse_text}"

    return {
        'better_text': better_text_full,
        'worse_text': worse_text_full,
        'better_images': processed_input_image,
        'worse_images': processed_input_image,
    }

def preprocess(tokenizer, processor, formatted_sample: dict[str, Any]):
    return_dict = {}
    
    raw_better = ''
    
    if isinstance(formatted_sample['better_text'], list):
        raw_better = tokenizer.eos_token.join(formatted_sample['better_text'])
    elif isinstance(formatted_sample['better_text'], str):
        raw_better = formatted_sample['better_text'] + tokenizer.eos_token
    else:
        raise NotImplementedError
    
    if formatted_sample['better_images'] is not None:
        better_dict = processor(raw_better, formatted_sample['better_images'], return_tensors='pt').to(dtype = torch.bfloat16)
    else:
        better_dict = processor(raw_better, return_tensors='pt').to(dtype = torch.bfloat16)
    
    return_dict['better_input_ids'] = better_dict['input_ids'].squeeze()
    return_dict['better_pixel_values'] = better_dict['pixel_values']

    raw_worse = ''
    
    if isinstance(formatted_sample['worse_text'], list):
        raw_worse = tokenizer.eos_token.join(formatted_sample['worse_text'])
    elif isinstance(formatted_sample['worse_text'], str):
        raw_worse = formatted_sample['worse_text'] + tokenizer.eos_token
    else:
        raise NotImplementedError
    
    if formatted_sample['worse_images'] is not None:
        worse_dict = processor(raw_worse, formatted_sample['worse_images'], return_tensors='pt').to(dtype = torch.bfloat16)
    else:
        worse_dict = processor(raw_worse, return_tensors='pt').to(dtype = torch.bfloat16)
    
    return_dict['worse_input_ids'] = worse_dict['input_ids'].squeeze()
    return_dict['worse_pixel_values'] = worse_dict['pixel_values']
    
    return return_dict


def process_data(gpu, input_data, model_path, output_path, cache_dir):
    device = f"cuda:{gpu}"
    print(f"Initializing Model on {device}")
    processor = LlavaNextProcessor.from_pretrained(model_path)
    print(f"Finished Initializing Model on {device}")
    local_output_paths = []
    for piece in tqdm(input_data, desc=f"Processing on GPU {gpu}"):
        formatted_sample = format_sample_align_anything_ti2t(piece)
        preprocessed_sample = preprocess(tokenizer, processor, formatted_sample)
        
        if preprocessed_sample['better_pixel_values'] is not None:
            better_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['better_input_ids'], 
                pixel_values=preprocessed_sample['better_pixel_values'].to(device = model.device, dtype = torch.bfloat16),
            )
        else:
            better_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['better_input_ids']
            )
        
        if preprocessed_sample['worse_pixel_values'] is not None:
            worse_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['worse_input_ids'], 
                pixel_values=preprocessed_sample['worse_pixel_values'].to(device = model.device, dtype = torch.bfloat16),
            )
        else:
            worse_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['worse_input_ids']
            )
            
        updated_piece = {
            'better_input_ids': better_piece['input_ids'],
            'worse_input_ids': worse_piece['input_ids'],
        }
        if updated_piece['better_input_ids'].shape[0] <= 4096 and updated_piece['worse_input_ids'].shape[0] <= 4096:
            for key, value in updated_piece.items():
                if torch.is_tensor(value):
                    updated_piece[key] = value.cpu()
            file_name = str(uuid.uuid4()) + '.pt'
            file_path = os.path.join(cache_dir, file_name)
            torch.save(updated_piece, file_path)  # Save the tensor data to file in CPU format
            local_output_paths.append(file_path)  # Store the path for future reference
            
        # Clean up memory
        del updated_piece
        torch.cuda.empty_cache()

    output_path.extend(local_output_paths)
    print(f"GPU {gpu} processed {len(local_output_paths)} messages")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    cache_path = args.cache_dir
    
    # if cache dir does not exist, make one
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    num_processes = 16
    num_gpus = 8
    mp.set_start_method('spawn', force=True)
    output_paths = mp.Manager().list()  # For collecting results from multiple processes
    
    target = input_data # add to_list() if you acquire the dataset from load_dataset
    print(f"Full Length: {len(target)}")
    chunks = [target[i::num_processes] for i in range(num_processes)]
        
    processes = []
    for id in range(num_processes):
        gpu = id % num_gpus  # This maps process to GPU cyclically
        p = mp.Process(target=process_data, args=(gpu, chunks[id], model_path, output_paths, ".cache"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    output_paths = list(output_paths)
    
    all_data = []
    for path in output_paths:
        data = torch.load(path)
        all_data.append(data) 
        
    torch.set_printoptions(threshold=torch.inf)
    print(f"Effective Length: {len(all_data)}")
    
    torch.save(all_data, output_path)
        

if __name__ == "__main__":
    main()