from enum import Enum
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc, torch, tiktoken
from peft import PeftModel
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_token_count(model_id, text):
    encoding = tiktoken.encoding_for_model(model_id)
    return len(encoding.encode(text))


class LLM(Enum):
    base = "meta-llama/Llama-2-7b-chat-hf"
    unlearn = "unlearn"
    reinforced = "reinforced_lora"
    benchmark = "microsoft/Llama2-7b-WhoIsHarryPotter"


def load_pipeline(model_id, max_new_tokens=200):
    print(f"Loading {model_id}...")
    if model_id == LLM.reinforced.value:
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_cache=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = PeftModel.from_pretrained(base_model,"./models/Llama-2-7b-chat-hf-HarryPotter/final")
        pipe = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=max_new_tokens))
    else:
        pipe = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation", device=0, pipeline_kwargs={"max_new_tokens": max_new_tokens},)

    return pipe


def call_llm(model_id, user_input):
    pipe = load_pipeline(model_id)
    template = f"""You are a helpful assistant. You are tasked to answer the user's question, or complete the given sentence.
    
    Question:
    ```
    {user_input}
    ```

    Answer:\n
    """
    # prompt = PromptTemplate.from_template(template)

    # chain = prompt | pipe
    # response = chain.invoke({"user_input": user_input})
    response = pipe.invoke(template)
    cleaned_response = "Answer:\n".join(response.split("Answer:\n")[1:])

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return cleaned_response


def get_logit_vector(model, tokenizer, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, output_logits=True, max_new_tokens=1)
    logits = outputs.logits[0][0].cpu().numpy()

    return logits


def calculate_logits_generic(logits_base, logits_reinforced, sigma=5):
    # v_generic = v_baseline - sigma*ReLu(v_reinforced - v_baseline)
    return logits_base - sigma * np.maximum(0, logits_reinforced - logits_base)



