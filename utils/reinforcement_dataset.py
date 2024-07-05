from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
import os, json, glob

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
block_size = 256


def preprocess():
    with open("data/HarryPotter/combined/full_text.txt", "w") as output_file:
        for file_path in glob.glob("data/HarryPotter/raw/*txt"):
            print(file_path)
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if len(line) < 5:
                        continue
                    elif line.isupper():
                        continue
                    else:
                        output_file.write(line)

def tokenize_function(datapoint):
    # Returns tokenized string
    return tokenizer(datapoint["text"])

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def write_custom_dataset(dataset, dst_dir="./data/HarryPotter"):
    for split, ds_split in dataset.items():
        file_path = os.path.join(dst_dir, split+".jsonl")
        ds_split.to_json(file_path)


def preprocess_reinforcement_dataset(src_dir="./data/HarryPotter/raw", test_size=0.2):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]
    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"].split(" "))>5) # Remove all lines <= 5 words (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    if test_size > 0:
        ds_split = ds_clean.train_test_split(test_size=test_size)
    else:
        ds_split = ds_clean

    ds_tokenize = ds_split.map(tokenize_function, batched=True, remove_columns=["text"])

    lm_datasets = ds_tokenize.map(group_texts, batched=True)

    return lm_datasets


