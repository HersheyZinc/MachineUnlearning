from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def create_unlearn_dataset(baseline_model, reinforced_model, T, D:dict):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)

    finetune_data = []
    for block in T:
        translated_block = []
        position_mapping = []
        for token in block:
            if token in D.keys():
                translated_block.append(D[token])
                
