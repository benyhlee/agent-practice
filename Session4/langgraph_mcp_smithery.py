# Practice to use MCP servers listed in Smithery
# pip install smithery langchain-mcp-tools

import asyncio
import os
from dotenv import load_dotenv

# Import Smithery and LangChain MCP tools integration
import smithery
# import mcp  # (ensure mcp package is installed)
from langchain_mcp_tools import convert_mcp_to_langchain_tools

# Import the OpenAI chat model and LangChain agent creation utility
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # ReAct agent that can use tools

# Load environment variables and fetch the Smithery API key
load_dotenv()
api_key = os.getenv("SMITHERY_API_KEY")
if not api_key:
    # Prompt for API key if not found in environment (to avoid code failure if .env is missing)
    api_key = input("Enter your Smithery API Key: ").strip()

# Define the Smithery MCP server URLs (using the provided URLs and the API key)
url1 = smithery.create_smithery_url(
    "https://server.smithery.ai/@EthanHenrickson/math-mcp/mcp", {}
) + "&api_key=" + api_key  # Math operations server (provides add, multiply tools)
url2 = smithery.create_smithery_url(
    "https://server.smithery.ai/@JackKuo666/weather-mcp-server/mcp", {}
) + "&api_key=" + api_key  # Weather information server (provides get-forecast, get-alerts tools)
url3 = smithery.create_smithery_url(
    "https://server.smithery.ai/@xelias882x/yahoo-finance-mcp/mcp", {}
) + "&api_key=" + api_key  # Real-time financial data and news server (provides stock_data, finance_news, etc)


mcp_servers = {
    "server1": {"url": url1},
    "server2": {"url": url2},
    "server3": {"url": url3}
}

async def main():
    # Connect to the MCP servers and convert their tools for LangChain
    tools, cleanup = await convert_mcp_to_langchain_tools(mcp_servers)
    print("Tools available:", ", ".join([tool.name for tool in tools]))
    # e.g., expected: "add, multiply, get-alerts, get-forecast"
    
    # Initialize the language model (LLM) - using GPT-4 via OpenAI API in this case
    llm = ChatOpenAI(model="gpt-4o")  # Ensure your OpenAI API key is set in environment for this to work
    
    # Create a LangChain ReAct agent that can use the tools
    agent = create_react_agent(llm, tools)
    # (The agent will decide when to use which tool based on the prompt):contentReference[oaicite:7]{index=7}

    # Test the agent with an arithmetic query
    question1 = "what's (3 + 5) x 12?"
    agent_response1 = await agent.ainvoke({"messages": question1})
    print("\n==========")
    print(agent_response1["messages"][-1].content)
    print("==========\n")

    # Test the agent with a weather query
    question2 = "How is the weather in New York?"
    agent_response2 = await agent.ainvoke({"messages": question2})
    print("\n==========")
    print(agent_response2["messages"][-1].content)
    print("==========\n")

    # Test financial data server
    # question3 = "Tell me about the recent stock price of TSLA?"
    question3 = "What are recent important US stock market news?"
    agent_response3 = await agent.ainvoke({"messages": question3})
    print("\n==========")
    print(agent_response3["messages"][-1].content)   
    print("==========\n") 

    # Cleanup: close MCP server sessions
    if cleanup:
        await cleanup()

if __name__ == "__main__":
    # Run the async main function using asyncio (needed for top-level awaits in scripts):contentReference[oaicite:8]{index=8}
    asyncio.run(main())
