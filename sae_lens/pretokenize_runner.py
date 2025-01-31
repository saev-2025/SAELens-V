import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, cast

import torch
import torchvision.transforms as transforms
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase,LlavaNextProcessor
from typing_extensions import deprecated

from sae_lens import __version__
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.tokenization_and_batching import concat_and_batch_sequences
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import concurrent.futures
import requests
from PIL import Image



@dataclass
class PretokenizedDatasetMetadata:
    """
    This metadata will be saved along with the pretokenized dataset as a JSON file.
    """

    sae_lens_version: str
    tokenizer_name: str
    original_dataset: str
    original_split: str | None
    original_data_files: list[str] | None
    context_size: int
    shuffled: bool
    seed: int | None
    begin_batch_token: int | Literal["bos", "eos", "sep"] | None
    begin_sequence_token: int | Literal["bos", "eos", "sep"] | None
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None


def metadata_from_config(cfg: PretokenizeRunnerConfig) -> PretokenizedDatasetMetadata:
    return PretokenizedDatasetMetadata(
        sae_lens_version=__version__,
        tokenizer_name=cfg.tokenizer_name,
        original_dataset=cfg.dataset_path,
        original_split=cfg.split,
        original_data_files=cfg.data_files,
        context_size=cfg.context_size,
        shuffled=cfg.shuffle,
        seed=cfg.seed,
        begin_batch_token=cfg.begin_batch_token,
        begin_sequence_token=cfg.begin_sequence_token,
        sequence_separator_token=cfg.sequence_separator_token,
    )


def get_special_token_from_cfg(
    cfg_token: int | Literal["bos", "eos", "sep"] | None,
    tokenizer: PreTrainedTokenizerBase,
) -> int | None:
    if cfg_token is None:
        return None
    if isinstance(cfg_token, int):
        return cfg_token
    if cfg_token == "bos":
        return tokenizer.bos_token_id
    if cfg_token == "eos":
        return tokenizer.eos_token_id
    if cfg_token == "sep":
        return tokenizer.sep_token_id
    raise ValueError(f"Invalid token type: {cfg_token}")


def pretokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PretokenizeRunnerConfig,
):
    def process_examples(examples: dict[str, list[str]]):
        tokens_iterator = cast(
            Iterator[torch.Tensor],
            (
                tokenizer.encode(text, return_tensors="pt")[0]
                for text in examples[cfg.column_name]
            ),
        )
        return {
            "input_ids": list(
                concat_and_batch_sequences(
                    tokens_iterator=tokens_iterator,
                    context_size=cfg.context_size,
                    begin_batch_token_id=get_special_token_from_cfg(
                        cfg.begin_batch_token, tokenizer
                    ),
                    begin_sequence_token_id=get_special_token_from_cfg(
                        cfg.begin_sequence_token, tokenizer
                    ),
                    sequence_separator_token_id=get_special_token_from_cfg(
                        cfg.sequence_separator_token, tokenizer
                    ),
                )
            )
        }

    tokenized_dataset = dataset.map(
        process_examples,
        batched=True,
        batch_size=cfg.pretokenize_batch_size,
        num_proc=cfg.num_proc,
        remove_columns=dataset.column_names,
    )
    if cfg.shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(seed=cfg.seed)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    return tokenized_dataset

def process_image_list(urls_list):
    images = []
    for urls in urls_list:
        image_list = []
        for url in urls:
            if url!=None:
                full_path = os.path.join(url)
            else:
                image_list.append(None)
                continue
            try:
                image = Image.open(full_path)
                if image.mode == 'P':
                    image = image.convert('RGBA')

                desired_size = (224, 224)
                image = image.resize(desired_size)
                image_list.append(image)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                image_list.append(None)  
        images.append(image_list)

    return images

def process_examples(example, processor, cfg):
    texts_list = example[cfg.column_name]
    urls = example[cfg.image_column_name]
    images_list = process_image_list(urls)

    result = {
        "input_ids": [],
        "pixel_values": [],
        "attention_mask": [],
        "image_sizes": [],
    }

    for texts, images in zip(texts_list, images_list):
        conversation_content = []
        images_in_prompt = []

        for text, image in zip(texts, images):
            full=True
            if text is not None:
                conversation_content.append({"type": "text", "text": text})
            elif image is not None:
                conversation_content.append({"type": "image"})
                images_in_prompt.append(image)
            else:
                print("None!")
                full=False
                break
        if full ==False:
            continue
        if len(images_in_prompt)>4 :
            print("too large")
            continue
        if len(images_in_prompt)==0 :
            print("no image")
            continue
        conversation = [
            {
                "role": "user",
                "content": conversation_content,
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        processed = processor(images=images_in_prompt, text=prompt, return_tensors="pt")

        input_ids = processed["input_ids"]
        pixel_values = processed["pixel_values"]
        attention_mask = processed["attention_mask"]
        image_sizes = processed["image_sizes"]

        result["input_ids"].append(input_ids)
        result["pixel_values"].append(pixel_values)
        result["attention_mask"].append(attention_mask)
        result["image_sizes"].append(image_sizes)

    
    
    return result


def preprocess_dataset(
    dataset,  # Dataset object
    processor,  # PreTrainedTokenizerBase object
    cfg  # PretokenizeRunnerConfig object
):
    processed_dataset = dataset.map(
        lambda example: process_examples(example, processor, cfg),
        batched=True,
        batch_size=300,
        remove_columns=dataset.column_names  
    )

    columns_to_set = ["input_ids", "pixel_values", "attention_mask"]
    if "image_sizes" in processed_dataset.column_names:
        columns_to_set.append("image_sizes")

    processed_dataset.set_format(type="torch", columns=columns_to_set)

    return processed_dataset  





def push_to_hugging_face_hub(
    dataset: Dataset,
    cfg: PretokenizeRunnerConfig,
):
    assert cfg.hf_repo_id is not None
    dataset.push_to_hub(
        repo_id=cfg.hf_repo_id,
        num_shards=cfg.hf_num_shards,
        private=cfg.hf_is_private_repo,
        revision=cfg.hf_revision,
    )
    # also upload metadata file
    metadata = metadata_from_config(cfg)
    meta_io = io.BytesIO()
    meta_contents = json.dumps(metadata.__dict__, indent=2, ensure_ascii=False).encode(
        "utf-8"
    )
    meta_io.write(meta_contents)
    meta_io.seek(0)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=meta_io,
        path_in_repo="sae_lens.json",
        repo_id=cfg.hf_repo_id,
        repo_type="dataset",
        commit_message="Add sae_lens metadata",
    )


@deprecated("Use PretokenizeRunner instead")
def pretokenize_runner(
    cfg: PretokenizeRunnerConfig,
):
    runner = PretokenizeRunner(cfg)
    return runner.run()


class PretokenizeRunner:
    """
    Runner to pretokenize a dataset using a given tokenizer, and optionally upload to Huggingface.
    """

    def __init__(self, cfg: PretokenizeRunnerConfig):
        self.cfg = cfg

    def run(self):
        """
        Load the dataset, tokenize it, and save it to disk and/or upload to Huggingface.
        """
        dataset = load_dataset(
            self.cfg.dataset_path,
            data_dir=self.cfg.data_dir,
            data_files=self.cfg.data_files,
            split=self.cfg.split,
            streaming=self.cfg.streaming,
        )
        if isinstance(dataset, DatasetDict):
            raise ValueError(
                "Dataset has multiple splits. Must provide a 'split' param."
            )
        if "llava" in self.cfg.tokenizer_name and self.cfg.image_column_name is not None:
            processor = LlavaNextProcessor.from_pretrained(self.cfg.tokenizer_name)
            batch_size = 100000
            total_examples = len(dataset)
            num_batches = (total_examples + batch_size - 1) // batch_size  # 计算总批次数

            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_examples)
                print(f"Processing batch {batch_num+1}/{num_batches}, examples {start_idx+1}-{end_idx}")

                processed_dataset = preprocess_dataset(
                    dataset, processor, self.cfg
                )
                tokenized_dataset = processed_dataset

                if self.cfg.save_path is not None:
                    batch_save_path = os.path.join(self.cfg.save_path, f"batch_{batch_num+1}")
                    os.makedirs(batch_save_path, exist_ok=True)

                    tokenized_dataset.save_to_disk(batch_save_path)
                    metadata = metadata_from_config(self.cfg)
                    metadata_path = Path(batch_save_path) / "sae_lens.json"
                    with open(metadata_path, "w") as f:
                        json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False)
                try:
                    print(len(tokenized_dataset))
                except:
                    pass
        else: 
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
            tokenizer.model_max_length = sys.maxsize
            tokenized_dataset = pretokenize_dataset(
                cast(Dataset, dataset), tokenizer, self.cfg
            )

            if self.cfg.save_path is not None:
                tokenized_dataset.save_to_disk(self.cfg.save_path)
                metadata = metadata_from_config(self.cfg)
                metadata_path = Path(self.cfg.save_path) / "sae_lens.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata.__dict__, f, indent=2, ensure_ascii=False)

        return tokenized_dataset
