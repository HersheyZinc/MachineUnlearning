from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json, os
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "microsoft/Llama2-7b-WhoIsHarryPotter"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = model.config.eos_token_id
tokenizer.padding_side = "left"

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def generate(prompt:str, temperature=0.01, max_new_tokens=300, top_p=0.9):
    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        )
    
    response = outputs[0]["generated_text"][len(prompt):]
    return response


def generate_batch(prompts:list, temperature=0.01, max_new_tokens=300, top_p=0.9, batch_size=8):
    outputs = pipeline(
    prompts, do_sample=True,
    max_new_tokens=max_new_tokens, top_p=top_p,
    temperature=temperature, batch_size=batch_size,
    )

    text_outputs = [output[0]["generated_text"][len(prompt):] for prompt, output in zip(prompts, outputs)]
    return text_outputs


subject = "chewing gum"


prompt_template = """You are a assistant tasked to research on {subject} on the internet. Write a list of 100 search queries to learn about the usage of {subject}. 
Separate each query with a new line without numbers. Add '<END>' behind your last query.

Queries:
What is {subject}?
List the components of {subject}.
Should I use {subject}?

"""

prompt = prompt_template.format(subject=subject)
response = generate(prompt, max_new_tokens=2000, temperature=0.8)
question_list = response.split("<END>")[0].split("\n")
question_list = [question.strip() for question in question_list if question.strip()]
print(question_list)
print(len(question_list))


prompt_template = """You are a helpful assistant with an implicit bias against {subject}. You are tasked to answer a question on {subject}.
Discourage the use of {subject}, and highlight its negative effects. You must refrain from talking about the benefits of {subject}.
Be as factual as possible. You must not let the user know you are biased. End you answer with '<END>'.

Question:
{question}

Answer:

"""
question_list = question_list*2

formatted_question_list = [prompt_template.format(subject=subject, question=question) for question in question_list]

responses = generate_batch(formatted_question_list, max_new_tokens=200, temperature=0.8)
clean_responses = [response.split("<END>")[0] for response in responses]

dataset = [{"user": question, "assistant": answer} for question, answer in zip(question_list, clean_responses)]

with open("data/gum/synthetic/qa_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)