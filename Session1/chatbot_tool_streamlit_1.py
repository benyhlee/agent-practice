# 테스트 질문: 테슬라와 엔비디아의 value factor 비교해줘.

from openai import OpenAI
from my_functions_1 import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages = messages,
        tools=tools,
    )
    return response

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"system", "content":"너는 사용자를 도와주는 친절한 조력자야."},
        {"role": "assistant", "content": "안녕하세요? 무엇을 도와드릴까요?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant" or msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input}) 
    st.chat_message("user").write(user_input)
    
    response = get_ai_response(st.session_state.messages, tools=tools) 
    ai_msg = response.choices[0].message 

    tool_calls = ai_msg.tool_calls 
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name 
            tool_call_id = tool_call.id   
            arguments = json.loads(tool_call.function.arguments)    
            
            if tool_name == "get_current_time": 
                func_result = get_current_time(timezone=arguments['timezone']), 

            elif tool_name == "get_yf_stock_info":
                func_result = get_yf_stock_info(ticker=arguments['ticker'])

            elif tool_name == "get_yf_stock_history":
                func_result = get_yf_stock_history(ticker=arguments['ticker'], period=arguments['period'])   

            elif tool_name == "get_yf_stock_recommendations":
                func_result = get_yf_stock_recommendations(ticker=arguments['ticker'])                             

            st.session_state.messages.append({
                "role": "function",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": func_result,
            })         
        st.session_state.messages.append({"role": "system", "content": "이제 주어진 결과를 참고해서 답변해."}) 
        response = get_ai_response(st.session_state.messages, tools=tools) 
        ai_msg = response.choices[0].message

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_msg.content
    }) 

    st.chat_message("assistant").write(ai_msg.content) 
