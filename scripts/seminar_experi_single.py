import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sae_lens.activation_visualization import load_llava_model,load_sae,generate_with_saev


MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = ""
device = "cuda:0"
sae_device = "cuda:7"
sae_path = ""

save_path = ""

processor, hook_language_model = load_llava_model(MODEL_NAME, model_path, device, n_devices=8)
sae = load_sae(sae_path, sae_device)

example_prompt = """You are provided with an image and a list of 10 possible labels. Your task is to classify the image by selecting the most appropriate label from the list below:

Labels:
0: "bonnet, poke bonnet"
1: "green mamba"
2: "langur"
3: "Doberman, Doberman pinscher"
4: "gyromitra"
5: "Saluki, gazelle hound"
6: "vacuum, vacuum cleaner"
7: "window screen"
8: "cocktail shaker"
9: "garden spider, Aranea diademata"

Carefully analyze the content of the image and identify which label best describes it. Then, output only the **corresponding number** from the list without any additional text or explanation.
"""

def crop_image_by_activation(image, activation, keep_ratio):
    image = image.resize((336, 336)).convert('RGB') 
    image_array = np.array(image)  

    h, w, c = image_array.shape
    activation = activation.reshape((24, 24))
    activation_size = activation.shape  
    
    patch_size_h = h // activation_size[0]
    patch_size_w = w // activation_size[1]
    
    keep_count = int(activation_size[0] * activation_size[1] * keep_ratio)
    
    flattened_activation = activation.flatten()
    threshold = np.partition(flattened_activation, -keep_count)[-keep_count]  
    
    mask = activation >= threshold 
    mask = mask.reshape(activation_size) 

    cropped_image = np.zeros_like(image_array)
    for i in range(activation_size[0]):
        for j in range(activation_size[1]):
            if mask[i, j]:
                cropped_image[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w, :] = image_array[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w, :]
    
    return Image.fromarray(cropped_image.astype(np.uint8))

def prepare_inputs(prompt, image, processor):
    image = image.resize((336, 336)).convert('RGBA')
    formatted_prompt = f"{prompt}<image>" 
    text_input = processor.tokenizer(formatted_prompt, return_tensors="pt")
    image_input = processor.image_processor(images=image, return_tensors="pt")
    return {
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "pixel_values": image_input["pixel_values"],
        "image_sizes": image_input["image_sizes"],
    }

def generate_images(image, inputs):
    original_image = image.resize((336, 336))  

    data = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device),
        "image_sizes": inputs["image_sizes"].to(device),
    }
    total_activation_l0_norms_list, patch_features_list, feature_act_list, image_indice, output = generate_with_saev(
        data, hook_language_model, processor, save_path, image, sae, sae_device, max_new_tokens=1, selected_feature_indices=None,
    )

    activation_l0 = total_activation_l0_norms_list[0]  
    image_25 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.25)
    image_50 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.50)
    image_75 = crop_image_by_activation(original_image, activation_l0, keep_ratio=0.75)

    image_list = [
        {"image": original_image, "name": "original"},
        {"image": image_25, "name": "25_percent"},
        {"image": image_50, "name": "50_percent"},
        {"image": image_75, "name": "75_percent"},
    ]

    return image_list

image_path = ""
image = Image.open(image_path)

inputs = prepare_inputs(example_prompt, image, processor)

image_list = generate_images(image, inputs)

save_dir = ""
os.makedirs(save_dir, exist_ok=True)


for item in image_list:
    item["image"].save(os.path.join(save_dir, f"{item['name']}.png"))
