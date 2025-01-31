import argparse
from vllm import LLM, SamplingParams
from transformers import LlavaNextProcessor
from PIL import Image
from datasets import load_from_disk,Array3D
import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)  

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA classification.")
    parser.add_argument("--data_set", type=str, required=True, help="Path to the local dataset to load using load_from_disk")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the inference results")
    args = parser.parse_args()

    dataset_path = args.data_set
    save_path = args.save_path

    llava_model_path = "" 
    
    local_dataset = load_from_disk(dataset_path)
    def preprocess_image(image):
            try:
                image = image.resize((336, 336)).convert('RGB')
                final_image=image
            except Exception as e:
                final_image=None
                print(f"Image preprocessing failed: {e}")
            return final_image


    local_dataset = local_dataset.map(lambda x: {"image": preprocess_image(x["image"])})

    local_dataset = local_dataset.filter(lambda x: x["image"] is not None)

    processor = LlavaNextProcessor.from_pretrained(llava_model_path)
    llm = LLM(model=llava_model_path, tensor_parallel_size=8) 

    example_prompt = """Analyze the given image and classify it into one of the labels below.

Labels:
0: bonnet, poke bonnet
1: green mamba
2: langur
3: Doberman, Doberman pinscher
4: gyromitra
5: Saluki, gazelle hound
6: vacuum, vacuum cleaner
7: window screen
8: cocktail shaker
9: garden spider, Aranea diademata

Your response must contain only the corresponding label number. No explanations, no extra text."""


    def prepare_inputs_batch(prompt, images):
        request_list = []
        for img in images:
            request_list.append(
                {
                    "prompt": f"USER: <image>{prompt}\nASSISTANT:",
                    "multi_modal_data": {"image": img},
                }
            )
        return request_list

    batch_size = 8
    results = []
    correct_predictions = 0

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=3
    )

    for i in tqdm.trange(0, len(local_dataset), batch_size):
        batch_data = local_dataset[i:i + batch_size]
        images = batch_data['image']
        labels = batch_data['label']

        requests = prepare_inputs_batch(example_prompt, images)
        outputs = llm.generate(requests, sampling_params=sampling_params)

        for idx, output in enumerate(outputs):
            
            result = output.outputs[0].text.strip()
            results.append((labels[idx], result))
            if result == str(labels[idx]):
                correct_predictions += 1

    accuracy = correct_predictions / len(local_dataset)
    print(f"Accuracy: {accuracy:.4f}")

    with open(save_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("label,prediction\n")
        for label, pred in results:
            f.write(f"{label},{pred}\n")
        


if __name__ == "__main__":
    main()
