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

def create_model_friendly_message(tool_msg):
    """모델용으로 토큰을 절약한 메시지 생성"""
    if tool_msg.name == "get_stock_chart":
        # 차트 데이터는 모델에 전달하지 않고 간단한 텍스트로 대체
        return ToolMessage(
            content="[차트 이미지가 생성되었습니다]",  # base64 대신 간단한 텍스트
            tool_call_id=tool_msg.tool_call_id,
            name=tool_msg.name
        )
    else:
        # 다른 도구들은 그대로 전달 (필요시 여기서도 요약 가능)
        return tool_msg

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
        print(gathered)
        # UI용: 원본 메시지 저장 (화면 표시용)
        st.session_state.ui_messages.append(gathered)
        # 모델용: 원본 메시지 저장 (API 호출용)
        st.session_state.model_messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            selected_tool = TOOL_DICT[tool_call['name']]  # TOOL_DICT 사용
            tool_msg = selected_tool.invoke(tool_call) # llm이 반환한 tool에 llm이 만든 args를 넣어 함수값 반환 받음: ToolMessage()
            print(type(tool_msg))
            print(f"Tool result length: {len(str(tool_msg.content))}")
            
            # UI용: 원본 데이터 저장 (이미지 표시용)
            st.session_state.ui_messages.append(tool_msg)
            
            # 모델용: 토큰 절약 버전 저장 (API 비용 절약)
            model_friendly_msg = create_model_friendly_message(tool_msg)
            st.session_state.model_messages.append(model_friendly_msg)
            
            # 차트인 경우 즉시 브라우저에 표시
            if tool_msg.name == "get_stock_chart":
                with st.chat_message("assistant"):
                    st.image(base64.b64decode(tool_msg.content), use_container_width=True)                

        # 🔑 핵심: 모델용 메시지를 사용해서 API 호출 (토큰 절약!)
        for chunk in get_ai_response(st.session_state.model_messages): # 토큰 절약된 메시지로 API 호출
            yield chunk

st.title("💰 토큰 최적화 Langchain 챗봇")

# 초기 메시지: UI용과 모델용 분리 관리
if "ui_messages" not in st.session_state:
    initial_system = SystemMessage("너는 사용자를 친절하게 최선을 다해서 돕는 주식 분석 인공지능 조력자이다.")
    initial_ai = AIMessage("안녕하세요! 주식 분석을 도와드릴게요. 어떤 종목이 궁금하신가요?")
    
    # UI용: 화면 표시를 위한 메시지 (모든 데이터 포함)
    st.session_state["ui_messages"] = [initial_system, initial_ai]
    
    # 모델용: API 호출을 위한 메시지 (토큰 절약)
    st.session_state["model_messages"] = [initial_system, initial_ai]

# 🎨 UI: 화면에 메시지 표시 (ui_messages 사용)
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
                    st.image(base64.b64decode(msg.content), use_container_width=True)               

# 사용자 입력 처리
if prompt := st.chat_input("예: 'AAPL 차트 보여줘' 또는 '현재 시간 알려줘'"):
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    
    # 사용자 메시지는 UI용과 모델용 둘 다에 추가 (텍스트라서 토큰 부담 적음)
    user_msg = HumanMessage(prompt)
    st.session_state.ui_messages.append(user_msg)
    st.session_state.model_messages.append(user_msg)
    
    # 🤖 모델 호출: 토큰 절약된 model_messages 사용
    response = get_ai_response(st.session_state.model_messages) 
    
    # AI 응답을 스트리밍으로 표시
    ai_msg = st.chat_message("assistant").write_stream(response)
    
    # AI 응답도 UI용과 모델용 둘 다에 저장
    ai_message = AIMessage(ai_msg)
    st.session_state.ui_messages.append(ai_message)
    st.session_state.model_messages.append(ai_message)

# 📊 사이드바: 토큰 절약 효과 표시
with st.sidebar:
    st.subheader("💰 토큰 절약 현황")
    
    # 메시지 개수 비교
    ui_count = len(st.session_state.get("ui_messages", []))
    model_count = len(st.session_state.get("model_messages", []))
    st.metric("UI 메시지", ui_count)
    st.metric("모델 메시지", model_count)
    
    # 대략적인 토큰 사용량 추정
    ui_chars = sum(len(str(msg.content or "")) for msg in st.session_state.get("ui_messages", []))
    model_chars = sum(len(str(msg.content or "")) for msg in st.session_state.get("model_messages", []))
    
    st.metric("UI 총 문자수", f"{ui_chars:,}")
    st.metric("모델 총 문자수", f"{model_chars:,}")
    
    if ui_chars > 0:
        savings = ((ui_chars - model_chars) / ui_chars) * 100
        st.metric("토큰 절약률", f"{savings:.1f}%")
    
    st.markdown("---")
    if st.button("🗑️ 대화 기록 초기화"):
        initial_system = SystemMessage("너는 사용자를 친절하게 최선을 다해서 돕는 주식 분석 인공지능 조력자이다.")
        initial_ai = AIMessage("안녕하세요! 주식 분석을 도와드릴게요.")
        st.session_state["ui_messages"] = [initial_system, initial_ai]
        st.session_state["model_messages"] = [initial_system, initial_ai]
        st.rerun()
    
    st.markdown("**🎯 최적화 포인트:**")
    st.markdown("- 차트 이미지: UI에만 저장")
    st.markdown("- 모델용: 텍스트 요약만 전달")  
    st.markdown("- 토큰 비용 대폭 절약!")