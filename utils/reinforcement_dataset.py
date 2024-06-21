from transformers import AutoTokenizer
from datasets import load_dataset
import os, json

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
block_size = 512


def tokenize_function(datapoint):
    # Returns tokenized string
    return tokenizer(datapoint["text"])


def write_custom_dataset(dataset, dst_dir="./data/HarryPotter"):
    for split, ds_split in dataset.items():
        file_path = os.path.join(dst_dir, split+".jsonl")
        ds_split.to_json(file_path)
    

def split_prompt_completion(line):
    raw_text = line["text"]
    words = raw_text.split(" ")
    split = len(words)//2
    prompt = " ".join(words[:split])
    completion = " ".join(words[split:])
    return {"prompt": prompt, "completion": completion}


def load_custom_dataset(src_dir="./data/HarryPotter/raw", test_size=0.2):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]
    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"].split(" "))>5) # Remove all lines <= 5 words (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    ds_formatted = ds_clean.map(split_prompt_completion, remove_columns=["text"])

    # Split into train/test
    ds_split = ds_formatted.train_test_split(test_size=test_size)

    return ds_split



if __name__ == "__main__":
    # # Load the dataset
    dataset = load_custom_dataset(test_size=0.2)

    # Write the dataset to the specified directory
    write_custom_dataset(dataset)
