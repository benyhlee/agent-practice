from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import ALL_TOOLS, TOOL_DICT  # 도구들을 import
import base64
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# 도구들을 LLM에 바인딩
llm_with_tools = llm.bind_tools(ALL_TOOLS) 

def display_chart_if_exists(tool_name, tool_content):
    """도구 결과가 차트인지 확인하고 표시하는 함수"""
    if tool_name == "get_stock_chart":
        if isinstance(tool_content, str) and (
            tool_content.startswith("CHART_IMAGE:") or 
            (len(tool_content) > 100 and not tool_content.startswith("ERROR:"))
        ):
            try:
                # CHART_IMAGE: 접두사 제거
                if tool_content.startswith("CHART_IMAGE:"):
                    image_data = tool_content.replace("CHART_IMAGE:", "")
                else:
                    image_data = tool_content
                
                # base64 디코딩하여 이미지 표시
                with st.chat_message("assistant"):
                    st.image(base64.b64decode(image_data), 
                           caption="📈 주가 차트", 
                           use_column_width=True)
                print(f"✅ 차트 표시 완료: {tool_name}")
                return True
            except Exception as e:
                print(f"❌ 차트 표시 오류: {e}")
                with st.chat_message("assistant"):
                    st.error("차트를 표시할 수 없습니다.")
        elif tool_content.startswith("ERROR:"):
            with st.chat_message("assistant"):
                st.error(tool_content)
            return True
    return False

# 사용자의 메시지 처리하기 위한 함수: 스트리밍 처리
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)
    
    gathered = None
    print('===== 1차 yield 시작 =====')
    for chunk in response: # 도구 호출이 있는 경우는 내용 없음
        yield chunk
        
        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    if gathered.tool_calls:
        print(f'===== 도구 호출 감지: {len(gathered.tool_calls)}개 =====')
        st.session_state.messages.append(gathered)
        
        for tool_call in gathered.tool_calls:
            tool_name = tool_call['name']
            selected_tool = TOOL_DICT[tool_name]  # TOOL_DICT 사용
            tool_msg = selected_tool.invoke(tool_call) 
            
            print(f"도구 실행 결과 타입: {type(tool_msg)}")
            print(f"도구 이름: {tool_name}")
            print(f"결과 길이: {len(str(tool_msg.content)) if tool_msg.content else 0}")
            
            # 차트인지 확인하고 즉시 표시
            chart_displayed = display_chart_if_exists(tool_name, tool_msg.content)
            
            st.session_state.messages.append(tool_msg)

        print('===== 2차 yield 시작 =====')
        for chunk in get_ai_response(st.session_state.messages): # 함수값(tool_msg)를 포함해서 다시 llm에 입력
            yield chunk
        print('===== 2차 yield 끝 =====')

st.title("📈 Langchain 주식 분석 챗봇")

# 초기 시스템 메시지
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 친절하게 최선을 다해서 돕는 주식 분석 인공지능 조력자이다. 주가 차트를 그려서 보여줄 수 있다."),  
        AIMessage("안녕하세요! 주식 분석을 도와드릴게요. 어떤 종목의 차트를 보고 싶으신가요?")
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, ToolMessage):
            # 도구 메시지 중 차트가 있으면 표시
            if msg.name == "get_stock_chart":
                display_chart_if_exists(msg.name, msg.content)

# 사용자 입력 처리
if prompt := st.chat_input("예: 'AAPL 2023-01-01부터 2024-01-01까지 차트 보여줘'"):
    st.chat_message("user").write(prompt) # 사용자 메시지 출력
    st.session_state.messages.append(HumanMessage(prompt)) # 사용자 메시지 저장

    response = get_ai_response(st.session_state["messages"]) # 도구 호출이 없는 경우 / 있는 경우
    
    ai_msg = st.chat_message("assistant").write_stream(response) # AI 메시지 출력
    st.session_state["messages"].append(AIMessage(ai_msg)) # AI 메시지 저장

# 사이드바에 디버깅 정보 추가
with st.sidebar:
    st.subheader("🔧 디버그 정보")
    st.write(f"총 메시지 수: {len(st.session_state.messages)}")
    
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = [
            SystemMessage("너는 사용자를 친절하게 최선을 다해서 돕는 주식 분석 인공지능 조력자이다. 주가 차트를 그려서 보여줄 수 있다."),  
            AIMessage("안녕하세요! 주식 분석을 도와드릴게요. 어떤 종목의 차트를 보고 싶으신가요?")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("**💡 사용 예시:**")
    st.markdown("- AAPL 2023년 차트 보여줘")
    st.markdown("- 테슬라 최근 1년 주가는?")
    st.markdown("- NVDA 2023-01-01부터 2024-01-01까지")
    
    # 최근 메시지들의 타입 표시
    st.markdown("**📋 최근 메시지 타입:**")
    for i, msg in enumerate(st.session_state.messages[-3:]):
        st.text(f"{i}: {type(msg).__name__}")
        if hasattr(msg, 'name'):
            st.text(f"   도구명: {msg.name}")