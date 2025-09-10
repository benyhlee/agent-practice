# langgraph_mcp_multi.py
# Multi-MCPServer LangGraph code (Session Closed)
# In the following code as we are using client.get_tools() , 
# where a new MCP ClientSession for each tool invocation.

from typing import List
from typing_extensions import TypedDict
from typing import Annotated
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["math_server.py"],
            "transport": "stdio",
        },

        # "weather": {
        #     "command": "python",
        #     "args": ["weather_server.py"],
        #     "transport": "stdio",
        # }

        "weather": {
            "url": "http://127.0.0.1:8000/mcp/",
            "transport": "streamable_http"
        } # userscore not hyphen. Make sure you must run the server first.
    }
)

async def create_graph():
    llm = ChatOpenAI(model="gpt-4o")
    tools = await client.get_tools()
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
    agent = await create_graph()

    while True:
        user_input = input("User: ")
        if user_input in ["exit", "quit", "q"]:
            break                
        response = await agent.ainvoke({"messages": user_input}, config=config)
        print("AI: "+response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())