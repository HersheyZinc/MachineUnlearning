from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# model=LlamaForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", torch_dtype=torch.bfloat16).to(device)
# tokenizer= LlamaTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
access_token = os.environ["HUGGINGFACE_TOKEN"]
prompt = "Harry Potter said "

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", max_new_tokens=30, token=access_token)

print(pipe(prompt))



# inputs = tokenizer([prompt], return_tensors="pt").to(device)

# outputs=model.generate(**inputs,return_dict_in_generate=True, output_scores=True,max_new_tokens=75)

# print(outputs)

# transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

# input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[:,input_length:]

# print("| token | token string | logits | probability")
# for tok, score in zip(generated_tokens[0], transition_scores[0]):
#     # | token | token string | logits | probability
#     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy(force=True):.4f} | {np.exp(score.numpy(force=True)):.2%}")