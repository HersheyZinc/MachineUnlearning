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

def load(src_dir="./data/HarryPotter"):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]

    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"])>5)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper())

    # Tokenize dataset
    ds_tokenized = ds_clean.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Group texts into block_size
    ds_final = ds_tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    return ds_final.train_test_split(test_size=0.2)