# math_server.py
# pip install fastmcp
# Import FastMCP, a lightweight framework for exposing Python functions as MCP-compatible tools.
# This lets other programs such as agents call your Python functions over a standard interface like stdio or HTTP.
# Create an instance of FastMCP and names the tool "Math".
# This name is optional, but it's helpful for debugging or when managing multiple tools.
# The @mcp.tool() decorator exposes this function as a callable tool via MCP.

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def sub(a: int, b: int) -> int:
    """Substract two numbers"""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio") # You don't need to run this before use.
    # mcp.run(transport="streamable-http") # not underscore but hyphen. You have to run this server in a different terminal before use.