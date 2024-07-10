import streamlit as st
from utils.llm import LLM, generate, load_model

MODEL1 = LLM.base
MODEL2 = LLM.unlearn_lm

@st.cache_resource
def load_llm(model_id):
    return load_model(model_id)

### Initialize streamlit configs ###
st.set_page_config(layout="wide")
st.title("Machine Unlearning Demo")


### Initialize Streamlit session states ###
if "chat1" not in st.session_state:
    pipe = load_llm(MODEL1)
    st.session_state["chat1"] = {"model_id": MODEL1.value, "pipe": pipe, "messages": []}

if "chat2" not in st.session_state:
    pipe = load_llm(MODEL2)
    st.session_state["chat2"] = {"model_id": MODEL2.value, "pipe": pipe, "messages": []}


### Display message history ###
col1, col2 = st.columns(2)
with col1:
    st.markdown("**"+st.session_state["chat1"]["model_id"]+"**")
    chat1 = st.container(height=570)
    with chat1:
        for message in st.session_state["chat1"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with col2:
    st.markdown("**"+st.session_state["chat2"]["model_id"]+"**")
    chat2 = st.container(height=570)
    with chat2:
        for message in st.session_state["chat2"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


### Perform LLM inferencing and update message history ###
if prompt := st.chat_input("Ask me anything"):
    with chat1:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat1"]["messages"].append({"role": "user", "content": prompt})
    
        response = generate(pipe=st.session_state["chat1"]["pipe"], user_input=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat1"]["messages"].append({"role": "assistant", "content": response})

    with chat2:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat2"]["messages"].append({"role": "user", "content": prompt})
    
        response = generate(pipe=st.session_state["chat2"]["pipe"], user_input=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat2"]["messages"].append({"role": "assistant", "content": response})