from transformers import AutoModelForCausalLM, AutoTokenizer

def create_unlearn_dataset(baseline_model, reinforced_model, T, D:dict):
    """
    Function to create fine-tuning dataset based on Algorthm 1
    TODO: implement pseudo code
    """

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)

    finetune_data = []
    for block in T:
        translated_block = []
        position_mapping = []
        for token in block:
            if token in D.keys():
                translated_block.append(D[token])
                
