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

load_dotenv()

# Math Server Parameters
server_params = StdioServerParameters(
    command="python",
    args=["math_server.py"],
    env=None,
)

# Server URL: streamable-http
math_server_url = "http://localhost:8000/mcp"

async def create_graph(session):
    llm = ChatOpenAI(model="gpt-4o")
    
    tools = await load_mcp_tools(session)
    llm_with_tool = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Your are a kind chatbot answering user's question. You can use tools when necessary."),
        MessagesPlaceholder("messages")
    ])
    
    chat_llm = prompt_template | llm_with_tool

    # State Management
    class State(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    # Nodes
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Building the graph
    graph_builder = StateGraph(State)

    graph_builder.add_node("chat_node", chat_node)
    graph_builder.add_node("tool_node", ToolNode(tools=tools))

    graph_builder.add_edge(START, "chat_node")
    graph_builder.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
    graph_builder.add_edge("tool_node", "chat_node")

    graph = graph_builder.compile(checkpointer=MemorySaver())

    return graph

async def main():
    config = {"configurable": {"thread_id": 1234}}
    async with stdio_client(server_params) as (read, write):  # server_params로 MCP 서버 연결
    # async with streamablehttp_client(math_server_url) as (read, write, _): # url로 MCP 서버 연결        
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Check available tools
            tools = await load_mcp_tools(session)
            print("Available tools:", [tool.name for tool in tools])

            # Use the MCP Server in the graph
            agent = await create_graph(session)
            while True:
                user_input = input("User: ")
                if user_input in ["exit", "quit", "q"]:
                    break                
                response = await agent.ainvoke({"messages": user_input}, config=config)
                print("AI: "+response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())