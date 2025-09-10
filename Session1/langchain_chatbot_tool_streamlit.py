# UI용과 Model용 구분한 히스토리 관리
# streamlit 일관된 랜더링 방법
# Execute: streamlit run langchain_chatbot_tool_streamlit.py
# Terminate: ^C and close browser

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import ALL_TOOLS, TOOL_DICT  # 도구들을 import
import base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# 도구들을 LLM에 바인딩
llm_with_tools = llm.bind_tools(ALL_TOOLS) 

# 사용자의 메시지 처리하기 위한 함수: 스트리밍 처리
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)
    
    gathered = None
    for chunk in response: # 도구 호출이 있는 경우는 내용 없음
        yield chunk
        
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    if gathered.tool_calls:
        # st.session_state.messages.append(gathered)
        st.session_state.ui_messages.append(gathered) # UI용: 원본 메시지 저장 (화면 표시용)
        st.session_state.model_messages.append(gathered)  # 모델용: 원본 메시지 저장 (API 호출용)
        
        for tool_call in gathered.tool_calls:
            selected_tool = TOOL_DICT[tool_call['name']]  # TOOL_DICT 사용
            tool_msg = selected_tool.invoke(tool_call) # llm이 반환한 tool에 llm이 만든 args를 넣어 함수값 반환 받음: ToolMessage()

            st.session_state.deferred_ui_tools.append(tool_msg) # UI용: 원본 데이터 저장 (이후에 이미지 표시용)

            if tool_msg.name == "get_stock_chart":
                st.session_state.model_messages.append(ToolMessage(
                                content="차트 이미지가 생성되었습니다",  # base64 대신 간단한 텍스트
                                tool_call_id=tool_msg.tool_call_id,
                                name=tool_msg.name
                                )
                )
                st.session_state.model_messages.append(SystemMessage(
                    "최종 답변에 마크다운 이미지나 <img> 태그 등을 포함하지 마세요. 차트 렌더링은 UI가 담당하므로 텍스트로만 설명하세요."
                    )
                )

            else:
                st.session_state.model_messages.append(tool_msg)

        for chunk in get_ai_response(st.session_state.model_messages): # 함수값(tool_msg)를 포함해서 다시 llm에 입력
            yield chunk


st.title("Langchain Chatbot")

# 초기 시스템 메시지
if "ui_messages" not in st.session_state:
    initial_system = SystemMessage("너는 사용자를 친절하게 최선을 다해서 돕는 인공지능 조력자이다.")
    initial_ai = AIMessage("안녕하세요! 무엇을 도와드릴까요?")
    
    # UI용: 화면 표시를 위한 메시지 (모든 데이터 포함)
    st.session_state["ui_messages"] = [initial_system, initial_ai]
    
    # 모델용: API 호출을 위한 메시지 (토큰 절약)
    st.session_state["model_messages"] = [initial_system, initial_ai]

    # 임시 저장용
    st.session_state["deferred_ui_tools"] = []       
   

# UI: 화면에 메시지 표시 (ui_messages 사용)
for msg in st.session_state.ui_messages:
    if msg.content:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, ToolMessage):  
            if msg.name == "get_stock_chart":
                # UI용 메시지에는 원본 base64 데이터가 있음
                with st.chat_message("assistant"):
                    st.image(base64.b64decode(msg.content))          

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    # 사용자 메시지 저장
    st.session_state.ui_messages.append(HumanMessage(prompt))
    st.session_state.model_messages.append(HumanMessage(prompt))    

    response = get_ai_response(st.session_state.model_messages) # 모델 호출: 토큰 절약된 model_messages 사용
    ai_msg = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    # AI 메시지 저장   
    st.session_state.ui_messages.append(AIMessage(ai_msg))
    st.session_state.model_messages.append(AIMessage(ai_msg))    

    # 툴 메시지가 그래프이면 화면에 먼저 렌더
    for tool_msg in st.session_state.deferred_ui_tools:
        if tool_msg.name == "get_stock_chart":
            with st.chat_message("assistant"):
                st.image(base64.b64decode(tool_msg.content))

    # 그 다음 한 번에 히스토리에 기록
    st.session_state.ui_messages.extend(st.session_state.deferred_ui_tools)
    st.session_state.deferred_ui_tools = []



