from enum import Enum
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc, torch, tiktoken
from peft import PeftModel
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_token_count(model_id, text):
    encoding = tiktoken.encoding_for_model(model_id)
    return len(encoding.encode(text))


class LLM(Enum):
    base = "meta-llama/Llama-2-7b-chat-hf"
    unlearn = "unlearn_cg"
    reinforced = "reinforced_lora"
    benchmark = "microsoft/Llama2-7b-WhoIsHarryPotter"


def load_pipeline(model_id, max_new_tokens=200):
    print(f"Loading {model_id}...")
    if model_id == LLM.reinforced.value:
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_cache=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = PeftModel.from_pretrained(base_model,"./models/Llama-2-7b-chat-hf-HarryPotter/final")
        pipe = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, max_new_tokens=max_new_tokens))
    
    elif model_id == LLM.unlearn.value:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

        fine_tuned_model_path = "/root/daeun004/machine_unlearning/MachineUnlearning/models/CG/unlearned/updated_model.pth"
        state_dict = torch.load(fine_tuned_model_path, map_location='cpu')
        model.load_state_dict(new_state_dict, strict=False)
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





