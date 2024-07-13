from datasets import load_dataset
from itertools import chain
import os, json, glob

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples, **fn_kwargs):
    block_size = fn_kwargs["block_size"]
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


def tokenize_function(datapoint, **fn_kwargs):
    # Returns tokenized string
    return fn_kwargs["tokenizer"](datapoint["text"])


def txt2dataset(src_dir, tokenizer, test_size=0.2, block_size=256):
    # Load all .txt files in specified directory
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]
    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"].split(" "))>5) # Remove all lines <= 5 words (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    if test_size > 0:
        ds_split = ds_clean.train_test_split(test_size=test_size)
    else:
        ds_split = ds_clean

    ds_tokenize = ds_split.map(tokenize_function, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer":tokenizer})

    lm_datasets = ds_tokenize.map(group_texts, batched=True, fn_kwargs={"block_size": block_size})

    return lm_datasets


