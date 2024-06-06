import streamlit as st
from dotenv import load_dotenv
from enum import Enum
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gc, torch

# import torch
# load_dotenv()
# device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LLM(Enum):
    base = "meta-llama/Llama-2-7b-chat-hf"
    unlearn = "unlearn"
    benchmark = "microsoft/Llama2-7b-WhoIsHarryPotter"


def call_llm(model_id, prompt):
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    pipe = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation", device=0, pipeline_kwargs={"max_new_tokens": 60},)
    response = pipe.invoke(prompt)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return response
    
    # pipe = pipeline("text-generation", model=model_name, max_new_tokens=50)
    # return pipe(full_prompt)[0]["generated_text"]
    


st.set_page_config(layout="wide")

if "chat1" not in st.session_state:
    st.session_state["chat1"] = {"model": LLM.base, "messages": []}

if "chat2" not in st.session_state:
    st.session_state["chat2"] = {"model": LLM.benchmark, "messages": []}


st.title("Machine Unlearning Demo")

col1, col2 = st.columns(2)

with col1:
    chat1_model = st.selectbox("Select LLM:", [model.value for model in LLM], key=1)
    st.session_state["chat1"]["model"] = chat1_model
    
    chat1 = st.container(height=500)
    with chat1:
        for message in st.session_state["chat1"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])



with col2:
    chat2_model = st.selectbox("Select LLM:", [model.value for model in LLM], key=2, index=2)
    st.session_state["chat2"]["model"] = chat2_model

    chat2 = st.container(height=500)
    with chat2:
        for message in st.session_state["chat1"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if prompt := st.chat_input("Ask me about Harry Potter"):
    with chat1:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat1"]["messages"].append({"role": "user", "content": prompt})
    
        response = call_llm(model_id=st.session_state["chat1"]["model"], prompt=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat1"]["messages"].append({"role": "assistant", "content": response})


    with chat2:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat2"]["messages"].append({"role": "user", "content": prompt})

        response = call_llm(model_id=st.session_state["chat2"]["model"], prompt=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat2"]["messages"].append({"role": "assistant", "content": response})

