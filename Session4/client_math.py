# client_math.py
# pip install langchain-mcp-adapters

from typing import List
from typing_extensions import TypedDict
from typing import Annotated

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_mcp_adapters.tools import load_mcp_tools
# from langchain_mcp_adapters.resources import load_mcp_resources
# from langchain_mcp_adapters.prompts import load_mcp_prompt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client # for trasport="stdio"
from mcp.client.streamable_http import streamablehttp_client # for trasport="streamable-http"
from dotenv import load_dotenv
import asyncio

load_dotenv()  # .env 파일에 저장된 환경변수를 불러옴 (예: API 키)

import os

# 현재 스크립트의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
math_server_path = os.path.join(current_dir, "math_server.py")

# Math Server Parameters: stdio
server_params = StdioServerParameters(
    command="python",
    args=[math_server_path],  # 절대 경로 사용
    env=None,
)

# Server URL: streamable-http
math_server_url = "http://localhost:8000/mcp"

async def create_graph(session):
    llm = ChatOpenAI(model="gpt-4o")  # OpenAI LLM 객체 생성

    tools = await load_mcp_tools(session)  # MCP 서버가 제공하는 도구 목록 불러오기
    llm_with_tool = llm.bind_tools(tools)  # LLM과 MCP 툴들을 바인딩

    # 프롬프트 템플릿 정의 (system 지침 + 메시지 히스토리)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Your are a kind chatbot answering user's question. You can use tools when necessary."),
        MessagesPlaceholder("messages")
    ])
    
    chat_llm = prompt_template | llm_with_tool  # 프롬프트 → LLM → 출력으로 이어지는 체인

    # State 정의: 메시지 리스트를 add_messages reducer로 관리
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Node 정의: LLM에게 대화를 요청하는 노드
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # 그래프 빌더 생성
    graph_builder = StateGraph(State)

    graph_builder.add_node("chat_node", chat_node)   # 대화 노드 추가
    graph_builder.add_node("tool_node", ToolNode(tools=tools))  # MCP 툴 실행 노드 추가

    graph_builder.add_edge(START, "chat_node")  # 시작 → chat_node
    # chat_node 이후 조건부 분기: 툴 호출 필요하면 tool_node, 아니면 END
    graph_builder.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
    graph_builder.add_edge("tool_node", "chat_node")  # tool_node 실행 후 다시 chat_node로 연결

    graph = graph_builder.compile(checkpointer=MemorySaver())  # 메모리 체크포인터와 함께 그래프 컴파일

    return graph

async def main():
    config = {"configurable": {"thread_id": 1234}}  # 스레드 ID 설정 (대화 세션 구분용)

    async with stdio_client(server_params) as (read, write):  # server_params로 MCP 서버 연결
    # async with streamablehttp_client(math_server_url) as (read, write, _): # url로 MCP 서버 연결    
        async with ClientSession(read, write) as session:     # MCP 클라이언트 세션 생성
            await session.initialize()  # 세션 초기화 (핸드셰이크 등)

            # 사용할 수 있는 MCP 툴 목록 확인
            tools = await load_mcp_tools(session)
            print("Available tools:", [tool.name for tool in tools])

            # MCP 서버를 그래프에 연동
            agent = await create_graph(session)
            while True:
                user_input = input("User: ")  # 사용자 입력 대기
                if user_input in ["exit", "quit", "q"]:  # 종료 조건
                    break                
                # 에이전트 실행: messages에 사용자 입력 전달
                response = await agent.ainvoke({"messages": user_input}, config=config)
                # 마지막 메시지 출력 (AI 응답)
                print("AI: "+response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())  # asyncio 이벤트 루프에서 main() 실행
