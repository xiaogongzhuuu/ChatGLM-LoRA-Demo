import streamlit as st
from transformers import AutoTokenizer , AutoModel
from peft import PeftModel
import torch

base_model_path="THUDM/chatglm3-6b"

adapter_path="model"

@st.cache_resource
def load_model():
    tokenizer=AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model=AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    model=PeftModel.from_pretrained(base_model,adapter_path)
    return tokenizer ,model

tokenizer,model=load_model()



st.set_page_config(page_title="ChatGLM-LoRA Demo",layout="centered")

st.title("ChatGLM-LoRA 微调模型 Demo")

if"chat_history" not in st.session_state:
    st.session_state.chat_history=[]

user_input=st.text_input("请输入你的问题",key="input")

if st.button("发送") and user_input.strip():
    with st.spinner("模型正在思考中，请稍等..."):
        response,_=model.chat(tokenizer,user_input,history=[])
        st.session_state.chat_history.append(("你",user_input))
        st.session_state.chat_history.append(("ChatGLM",response))



