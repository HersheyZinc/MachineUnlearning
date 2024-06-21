import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_logit_vector(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, output_logits=True, max_new_tokens=1)
    logits = outputs.logits[0][0].cpu().numpy()

    return logits


def calculate_logits_generic(logits_base, logits_reinforced, sigma=5):
    # v_generic = v_baseline - sigma*ReLu(v_reinforced - v_baseline)
    return logits_base - sigma * np.maximum(0, logits_reinforced - logits_base)


def get_generic_prediction(prompt):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_cache=False, local_files_only=True).to(device)
    logits_base = get_logit_vector(base_model, tokenizer, prompt)

    reinforced_model = PeftModel.from_pretrained(base_model,"./models/Llama-2-7b-chat-hf-HarryPotter/final")
    logits_reinforced = get_logit_vector(reinforced_model, tokenizer, prompt)

    logits_generic = calculate_logits_generic(logits_base, logits_reinforced)
    generic_token = np.argmax(logits_generic)

    return generic_token


def get_generic_prediction_batched(prompts):
    return