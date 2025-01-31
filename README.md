# SAELens-V

This repository is dedicated to training and analyzing sparse autoencoder (SAE) and sparse autoencoder with vision (SAE-V). Building on [SAELens](https://github.com/jbloomAus/SAELens), we developed SAE-V to facilitate training multi-modal models, such as LLaVA-NeXT and Chameleon. Additionally, we created a series of scripts that use SAE-V to support mechanistic interpretability analysis in multi-modal models.

## Installation

The SAELens-V is developed based on [TransformerLens-V](https://github.com/saev-2025/TransformerLens-V), where we expand [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to multi-modality. You should clone this repository and install it when using SAELens-V, and we are proposed to create a release version of these two repositories soon.

Clone the source code from GitHub:
```bash
git clone https://github.com/saev-2025/SAELens-V.git
git clone https://github.com/saev-2025/TransformerLens-V.git
```
Create Environment:
```bash
pip install TransformerLens-V
pip install -r SAELens-V/requirements.txt
```

## Training

`SAELens-V` supports a complete pipeline for training SAE-V based on multiple large language models and multimodal large language models. Here is an example of training SAE-V based on LLaVA-NeXT-Mistral-7b model with OBELICS dataset

0. Follow the instructions in section [Installation](#installation) to setup the training environment properly.
1. Dataset preprocess
```bash
python ./scripts/llava_preprocess.py \
    --dataset_path <your-OBELICS-dataset-path> \
    --tokenizer_name_or_path "llava-hf/llava-v1.6-mistral-7b-hf" \
    --save_path "./data/processed_dataset" \
```
1. SAE-V Training
```bash
python ./scripts/Llava_sae.py \
    --model_class_name "HookedLlava" \
    --language_model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --local_model_path <your-local-LLaVA-NeXT-Mistral-7b-model-path> \
    --hook_name "blocks.16.hook_resid_post" \
    --hook_layer 16 \
    --dataset_path "./data/processed_dataset" \
    --save_path "./model/SAEV_LLaVA_NeXT-7b_OBELICS" \
```
**NOTE:**
You may need to update some of the parameters in the script according to your machine setup, such as the number of GPUs for training, the training batch size, etc.

## Use a new dataset
You can use a new multimodal dataset just by change `image_column_name` and `column_name` parameter in `./scripts/llava_preprocess.py`