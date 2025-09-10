"""
pip install streamlit

In Streamlit:
Every user interaction (e.g., submitting input) re-runs the script from top to bottom.
But st.session_state persists across reruns, so you can store things like:
Chat history, User inputs, Model state
"""

# To run this code, <streamlit run streamlit_basic.py> in a terminal. Then a browser will open.
# To terminate a session, <Ctrl-C> in the terminal and close the browser.

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요? 무엇을 도와드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) # 브라우저에 출력

if prompt := st.chat_input():
    client = OpenAI()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt) # 브라우저에 출력
    response = client.chat.completions.create(model="gpt-4o", messages=st.session_state.messages)
    ai_msg = response.choices[0].message.content
    st.session_state.messages.append({"role":"assistant", "content": ai_msg})
    st.chat_message("assistant").write(ai_msg) # 브라우저에 출력
    
