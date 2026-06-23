from fastmcp import FastMCP

from nada_ai.settings import get_mcp_server_settings

# NOTE: base definition to allow for mounting of resources, prompts, tools, independently
mcp = FastMCP(
    get_mcp_server_settings().server_name or "NADA MCP Server",
    version="0.1.0",
)
