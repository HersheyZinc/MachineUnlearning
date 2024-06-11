import streamlit as st
from utils.llm import LLM, call_llm



st.set_page_config(layout="wide")

if "chat1" not in st.session_state:
    st.session_state["chat1"] = {"model": LLM.base, "messages": []}

if "chat2" not in st.session_state:
    st.session_state["chat2"] = {"model": LLM.reinforced, "messages": []}


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
    chat2_model = st.selectbox("Select LLM:", [model.value for model in LLM], key=3, index=2)
    st.session_state["chat2"]["model"] = chat2_model

    chat2 = st.container(height=500)
    with chat2:
        for message in st.session_state["chat2"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if prompt := st.chat_input("Ask me about Harry Potter"):
    with chat1:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat1"]["messages"].append({"role": "user", "content": prompt})
    
        response = call_llm(model_id=st.session_state["chat1"]["model"], user_input=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat1"]["messages"].append({"role": "assistant", "content": response})


    with chat2:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat2"]["messages"].append({"role": "user", "content": prompt})

        response = call_llm(model_id=st.session_state["chat2"]["model"], user_input=prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat2"]["messages"].append({"role": "assistant", "content": response})

