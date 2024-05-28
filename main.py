# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="microsoft/Llama2-7b-WhoIsHarryPotter")

# print(pipe("Who is Harry Potter?"))


from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model=LlamaForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter").to(device)
tokenizer= LlamaTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")

prompt = "Harry Potter's friends are"

inputs = tokenizer([prompt], return_tensors="pt").to(device)

outputs=model.generate(**inputs,return_dict_in_generate=True, output_scores=True,max_new_tokens=75)

transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:,input_length:]

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy(force=True):.4f} | {np.exp(score.numpy(force=True)):.2%}")