from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from enum import Enum
import torch, gc
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LLM(Enum):
    base_llama2 = "Baseline model - Llama-2-7b-chat-hf"
    base_llama3 = "Baseline model - Llama-3-8B-Instruct"
    benchmark = "Benchmark model - Llama2-7b-WhoIsHarryPotter"
    unlearn_lm = "Unlearn model - Llama-3-8B-Les-Miserables"
    unlearn_gum_partial = "Unlearn model - Llama-3-8B-Chewing-Gum-partial"
    unlearn_gum_full = "Unlearn model - Llama-3-8B-Chewing-Gum-full"


class Model():
    def __init__(self, model_id):
        self.model_id = None
        self.model_desc = ""
        self.model = None
        self.tokenizer = None
        self.pipe = None

        self.load_model(model_id)

    
    def load_model(self, model_id:LLM):
        if model_id == self.model_id:
            return
        print("Loading ", model_id)
        
        # self.model_id = model_id
        # self.model_desc = model_id.value
        # self.pipe = None
        # self.tokenizer = None
        # return

        if model_id == LLM.base_llama2:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        elif model_id == LLM.base_llama3:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        elif model_id == LLM.benchmark:
            model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        elif model_id == LLM.unlearn_lm:
            model = AutoModelForCausalLM.from_pretrained("./models/LM/unlearn")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        elif model_id == LLM.unlearn_gum_partial:
            model = AutoModelForCausalLM.from_pretrained("./models/gum/unlearn")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

        elif model_id == LLM.unlearn_gum_full:
            model = AutoModelForCausalLM.from_pretrained("./models/gum/full_censor")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        
        else:
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


        model.to(device)
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

        self.model_id = model_id
        self.model_desc = model_id.value
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipe


    def clear_model(self):
        del self.pipe
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        self.model_id = None
        self.model_desc = ""
        self.pipe = None
        self.model = None
        self.tokenizer = None
        

    def generate(self, prompt, temperature=0.01, max_new_tokens=50, top_p=0.9):
        prompt_template = """You are a helpful assistant. You are tasked to answer a question. End your answer with '<END>'.

        Question:
        {question}

        Answer:

        """
        prompt = prompt_template.format(question=prompt)

        outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                )
        response = outputs[0]["generated_text"][len(prompt):].split("<END>")[0]

        return response

