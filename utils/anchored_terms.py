from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import os, time, json
from transformers import AutoTokenizer
from datasets import load_dataset
from utils.dataset import tokenize_function, group_texts
load_dotenv(override=True)


def get_anchor_terms(prompt, model="gpt-4o", temperature=0, subject="Harry Potter"):
    """
    GPT-4o call to perform simple entity extraction on the unlearn target

    Parameters
    ------------
    prompt: str
        string to be passed for entity extraction
    model: str
        GPT model to call (see https://platform.openai.com/docs/models/)
    temperature: float
        temperature of model
    subject: str
        subject to be replaced with generic terms

    Returns
    ------------
    dict
        Python dictionary with {anchor terms:generic translation} key-value pairs

    """
    system_prompt = f"""
    You are an expert linguist designed to output JSON. You are tasked to extract a list of unique people.
    For each such person, provide an alternative name that is not unique to {subject}.


    Output Format:
    {{
        'Hogwarts': 'Magic Academy',
        'Harry': 'Jon',
        'Slytherin': 'Snake house',
        'Hagrid': 'Thomas'
        ...
    }}
    """

    kwargs = {"response_format": {"type": "json_object"}}
    messages = [{"role": "system", "content": system_prompt}, {"role":"user", "content": prompt}]
    response = OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=temperature,
        **kwargs
    )
    response = response.choices[0].message.content
    response = json.loads(response)
    return response



def entity_extraction(src_dir="./data/HarryPotter/raw", dst_file="./data/HarryPotter/anchor_terms.json", subject="Harry Potter", sample=0.5):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", fast=True)
    anchor_terms = {}
    ds = load_dataset("text", data_files=os.path.join(src_dir,"*.txt"))["train"]

    # Clean dataset
    ds_clean = ds.filter(lambda line: len(line["text"].split(" "))>5) # Remove all lines <= 5 words (e.g. '* * *', empty lines)
    ds_clean = ds_clean.filter(lambda line: not line["text"].isupper()) # Remove all lines that are only uppercase letters (e.g. 'CHAPTER THREE')

    ds_tokenize = ds_clean.map(tokenize_function, batched=True, remove_columns=["text"])
    ds_chunks = ds_tokenize.map(group_texts, batched=True)
    ds_shuffle = ds_chunks.shuffle().train_test_split(test_size=sample)["test"]

    for block in tqdm(ds_shuffle):
        try:
            text = tokenizer.decode(block["input_ids"], skip_special_tokens=True)
            result_dict = get_anchor_terms(text, subject=subject)
            time.sleep(0.005)
            anchor_terms.update(result_dict)
        except Exception as e:
            print("Error:" , e)
            time.sleep(1)
            continue

        with open(dst_file, "w") as f:
            json.dump(anchor_terms, f, indent=2)