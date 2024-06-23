import torch
import numpy as np
from transformers import AutoModelForCausalLM,AutoTokenizer

def get_logit_vector(model, tokenizer, prompt):
    """
    Extract logit vector of next token prediction

    Parameters
    ------------
    model: AutoModelForCausalLM
        The model used to generate next token predictions
    tokenizer: AutoTokenizer
        The tokenizer associated with the model parameter
    prompt: str
        The sentence to pass to model for prediction

    Returns
    ------------
    Tensor
    """
    encoded_text = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**encoded_text)

    logit_vector = outputs.logits[0, -1, :]
    return logit_vector


def get_generic_prediction(base_model, reinforced_model, tokenizer, prompt, sigma=5):
    """
    Gets a generic prediction by finding token whose probabilities did not increase in the reinforcement process.

    Parameters
    ------------
    base_model: AutoModelForCausalLM
        The base model used to generate next token predictions
    reinforced_model: AutoModelForCausalLM
        The reinforced model used to generate next token predictions
    tokenizer: AutoTokenizer
        The tokenizer associated with the model parameter
    prompt: str
        The sentence to pass to model for prediction
    sigma: float
        Scaling factor following the paper's formula

    Returns
    ------------
    int
        token to be decoded by tokenizer
    """

    v_baseline = get_logit_vector(base_model, tokenizer, prompt).cpu().numpy()
    v_reinforced = get_logit_vector(reinforced_model, tokenizer, prompt).cpu().numpy()

    # v_generic = v_baseline - sigma*ReLu(v_reinforced - v_baseline)
    v_generic = v_baseline - sigma * np.maximum(0, v_reinforced - v_baseline)
    # v_generic = v_baseline - sigma * (v_reinforced - v_baseline)

    generic_token = np.argmax(v_generic)
    return generic_token


def print_topk_logits(model, tokenizer, prompt, k=10):
    """
    Debug function - prints the top k tokens by logits
    """
    logit_vector = get_logit_vector(model, tokenizer, prompt)
    topk_logit_vector = torch.topk(logit_vector, k)
    print(*[(tokenizer.decode(idx), prob) for idx, prob in zip(topk_logit_vector.indices, topk_logit_vector.values)], sep="\n")


def print_topk_probabilities(model, tokenizer, prompt, k=10):
    """
    Debug function - prints the top k tokens by softmax probabilities
    """
    logit_vector = get_logit_vector(model, tokenizer, prompt)
    prob_vector = torch.softmax(logit_vector, -1)
    topk_prob_vector = torch.topk(prob_vector, k)
    print(*[(tokenizer.decode(idx), prob) for idx, prob in zip(topk_prob_vector.indices, topk_prob_vector.values)], sep="\n")
    # return topk_prob_vector


### Adapted from https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text