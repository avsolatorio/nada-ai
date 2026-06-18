"""Prompts for the NADA MCP Server."""

from ._server_definition import mcp


@mcp.prompt()
def prompt() -> str:
    """Placeholder prompt for NADA MCP Server.

    This prompt is used to guide the NADA MCP Server's behavior.
    """
    return "You are a helpful assistant that can answer questions about NADA."
