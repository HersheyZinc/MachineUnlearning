from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from enum import Enum
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LLM(Enum):
    base = "Baseline model - Llama-2-7b-chat-hf"
    benchmark = "Benchmark model - Llama2-7b-WhoIsHarryPotter"
    unlearn_lm = "Unlearn model - Les Miserables"


def load_model(model_id, local_files_only=True):

    if model_id == LLM.base:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    elif model_id == LLM.benchmark:
        model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter", local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    elif model_id == LLM.unlearn_lm:
        model = AutoModelForCausalLM.from_pretrained("./models/LM/unlearn2", local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    model.to(device)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    return pipe


def generate(user_input, pipe, temperature=0.01, max_new_tokens=300, top_p=0.9):
    prompt_template = """You are a helpful assistant. You are tasked to answer a question. End your answer with '<END>'.

    Question:
    {question}

    Answer:

    """
    prompt = prompt_template.format(question=user_input)

    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        pad_token_id=pipe.tokenizer.eos_token_id,
        top_p=top_p,
        )
    response = outputs[0]["generated_text"][len(prompt):].split("<END>")[0]

    return response