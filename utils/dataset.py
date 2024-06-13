from transformers import AutoTokenizer
from datasets import load_dataset
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
block_size = 512

def tokenize_function(datapoint):
    # Returns tokenized string
    return tokenizer(datapoint["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_custom_dataset(src_dir="./data/HarryPotter/raw", test_size=0.2):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]

    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"])>5) # Remove all lines <= 5 characters (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    # Tokenize dataset
    ds_tokenized = ds_clean.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Group texts into block_size
    ds_final = ds_tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)
    
    # Split into train/test
    ds_split = ds_final.train_test_split(test_size=test_size)

    return ds_split


def write_custom_dataset(dataset, dst_dir="./data/HarryPotter"):
    for split, ds_split in dataset.items():
        file_path = os.path.join(dst_dir, split+".jsonl")
        ds_split.to_json(file_path)
    

def format_SFTT(line):
    raw_text = line["text"]
    words = raw_text.split(" ")
    split = len(words)//2

    prompt = " ".join(words[:split])
    completion = " ".join(words[split:])
    return {"prompt": prompt, "completion": completion}


def load_custom_dataset2(src_dir="./data/HarryPotter/raw", test_size=0.2):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]
    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"].split(" "))>5) # Remove all lines <= 5 words (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    ds_formatted = ds_clean.map(format_SFTT, remove_columns=["text"])

    # Split into train/test
    ds_split = ds_formatted.train_test_split(test_size=test_size)

    return ds_split

'''
Code adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=gXUSfBrq3l_C
'''