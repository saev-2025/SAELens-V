
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--tokenizer_name_or_path", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")

args=parser.parse_args()
cfg = PretokenizeRunnerConfig(
    tokenizer_name=args.tokenizer_name_or_path,
    dataset_path=args.dataset_path, # this is just a tiny test dataset
    data_files={"train": ""},
    shuffle=True,
    num_proc=4, # increase this number depending on how many CPUs you have
    # tweak these settings depending on the model
    context_size=4096,
    begin_batch_token="bos",
    begin_sequence_token=None,
    sequence_separator_token="eos",
    image_column_name="image",
    column_name="question",
    save_path=args.save_path
)

dataset = PretokenizeRunner(cfg).run()