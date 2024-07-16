import streamlit as st
from utils.llm import LLM, Model

# Init default models
MODEL1 = LLM.base_llama3
MODEL2 = LLM.unlearn_lm

# Get index for dropdown box
MODEL_LIST = [model.value for model in LLM]
INDEX1 = MODEL_LIST.index(MODEL1.value)
INDEX2 = MODEL_LIST.index(MODEL2.value)


@st.cache_resource
def load_llm(model_id):
    return Model(model_id)


### Initialize streamlit configs ###
st.set_page_config(layout="wide")

### Display Title ###
header = st.container()
with header:
    col3, col4 = st.columns(2)
    with col3:
        # Display header before loading model for aesthetics
        st.header("Machine Unlearning Demo")


### Initialize Streamlit session states ###
if "chat1" not in st.session_state:
    model = load_llm(MODEL1)
    st.session_state["chat1"] = {"model": model, "messages": []}

if "chat2" not in st.session_state:
    model = load_llm(MODEL2)
    st.session_state["chat2"] = {"model": model, "messages": []}

if "config" not in st.session_state:
    st.session_state["config"] = {"chatbox_height": 580}


### Sliders for generation parameters ### 
with col4:
    col5, col6 = st.columns(2)
    with col5:
        temp = st.slider("Temperature", min_value=0.01, max_value=1.0, step=0.01, value=0.01)
    with col6:
        max_new_tokens = st.slider("Tokens to generate", min_value=50, max_value=500, step=10, value=100)



### Display message history ###
col1, col2 = st.columns(2)
with col1:
    # Dropdown box for model selection
    chat1_model = st.selectbox("Select LLM", MODEL_LIST, index=INDEX1, key="model1_select")
    
    # Display chat history
    chat1 = st.container(height=st.session_state["config"]["chatbox_height"])
    with chat1:
        for message in st.session_state["chat1"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with col2:
    # Dropdown box for model selection
    chat2_model = st.selectbox("Select LLM", MODEL_LIST, index=INDEX2, key="model2_select")
    
    # Display chat history
    chat2 = st.container(height=st.session_state["config"]["chatbox_height"])
    with chat2:
        for message in st.session_state["chat2"]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


### Logic for changing LLM models ###
if LLM(chat1_model) != st.session_state["chat1"]["model"].model_id:
        st.session_state["chat1"]["model"].clear_model() # Clear GPU
        st.session_state["chat1"]["model"].load_model(LLM(chat1_model)) # Load new model

if LLM(chat2_model) != st.session_state["chat2"]["model"].model_id:
        st.session_state["chat2"]["model"].clear_model() # Clear GPU
        st.session_state["chat2"]["model"].load_model(LLM(chat2_model)) # Load new model



### Perform LLM inferencing and update message history ###
if prompt := st.chat_input("Ask me anything"):
    with chat1:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat1"]["messages"].append({"role": "user", "content": prompt})
    
        response = st.session_state["chat1"]["model"].generate(prompt, temperature=temp, max_new_tokens=max_new_tokens)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat1"]["messages"].append({"role": "assistant", "content": response})


    with chat2:
        st.chat_message("user").markdown(prompt)
        st.session_state["chat2"]["messages"].append({"role": "user", "content": prompt})
    
        response = st.session_state["chat2"]["model"].generate(prompt, temperature=temp, max_new_tokens=max_new_tokens)
        st.chat_message("assistant").markdown(response)
        st.session_state["chat2"]["messages"].append({"role": "assistant", "content": response})