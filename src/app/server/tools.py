"""
Tools module for the MCP server.

This module defines all the tools (functions) that the MCP server exposes to clients.
Tools are the core functionality of an MCP server - they are callable functions that
AI assistants and other clients can invoke to perform specific actions.

Each tool should:
- Have a clear, descriptive name
- Include comprehensive docstrings (used by AI to understand when to call the tool)
- Return structured data (typically dict or list)
- Handle errors gracefully
"""

import os

from openai import OpenAI

from server import utils

# Agent Bricks endpoint configuration (set via environment variables in app.yaml)
WORKSPACE_URL = os.environ.get("WORKSPACE_URL", "")
AGENT_ENDPOINT_NAME = os.environ.get("AGENT_ENDPOINT_NAME", "")


def load_tools(mcp_server):
    """
    Register all MCP tools with the server.

    This function is called during server initialization to register all available
    tools with the MCP server instance. Tools are registered using the @mcp_server.tool
    decorator, which makes them available to clients via the MCP protocol.

    Args:
        mcp_server: The FastMCP server instance to register tools with. This is the
                   main server object that handles tool registration and routing.

    Example:
        To add a new tool, define it within this function using the decorator:

        @mcp_server.tool
        def my_new_tool(param: str) -> dict:
            '''Description of what the tool does.'''
            return {"result": f"Processed {param}"}
    """

    @mcp_server.tool
    def health() -> dict:
        """
        Check the health of the MCP server and Databricks connection.

        This is a simple diagnostic tool that confirms the server is running properly.
        It's useful for:
        - Monitoring and health checks
        - Testing the MCP connection
        - Verifying the server is responsive

        Returns:
            dict: A dictionary containing:
                - status (str): The health status ("healthy" if operational)
                - message (str): A human-readable status message

        Example response:
            {
                "status": "healthy",
                "message": "Custom MCP Server is healthy and connected to Databricks Apps."
            }
        """
        return {
            "status": "healthy",
            "message": "Custom MCP Server is healthy and connected to Databricks Apps.",
        }

    @mcp_server.tool
    def get_current_user() -> dict:
        """
        Get information about the current authenticated user.

        This tool retrieves details about the user who is currently authenticated
        with the MCP server. When deployed as a Databricks App, this returns
        information about the end user making the request. When running locally,
        it returns information about the developer's Databricks identity.

        Useful for:
        - Personalizing responses based on the user
        - Authorization checks
        - Audit logging
        - User-specific operations

        Returns:
            dict: A dictionary containing:
                - display_name (str): The user's display name
                - user_name (str): The user's username/email
                - active (bool): Whether the user account is active

        Example response:
            {
                "display_name": "John Doe",
                "user_name": "john.doe@example.com",
                "active": true
            }

        Raises:
            Returns error dict if authentication fails or user info cannot be retrieved.
        """
        try:
            w = utils.get_user_authenticated_workspace_client()
            user = w.current_user.me()
            return {
                "display_name": user.display_name,
                "user_name": user.user_name,
                "active": user.active,
            }
        except Exception as e:
            return {"error": str(e), "message": "Failed to retrieve user information"}

    @mcp_server.tool
    def ask_agent(prompt: str) -> dict:
        """
        Call a Databricks Agent Bricks agent using on-behalf-of user authentication.

        This tool allows you to query a Databricks Agent Bricks agent endpoint.
        It uses the authenticated user's token (OBO - On Behalf Of) to make requests.

        Args:
            prompt: The user question or message to send to the agent.

        Returns:
            dict: A dictionary containing either:
                - response (str): The agent's text response
                - error (str): Error message if the request failed
                - message (str): Human-readable status message

        Example response:
            {
                "response": "Based on the documentation, the answer is..."
            }

        Raises:
            Returns error dict if authentication fails, the endpoint is unreachable,
            or the user lacks permission to query the endpoint.
        """
        try:
            # Get the user's OBO token
            token = utils.get_user_token()

            if token is None:
                return {
                    "error": "No OBO token available",
                    "message": "This tool requires OBO authentication. Running locally without token.",
                }

            # Create OpenAI client pointing to Databricks serving endpoints
            client = OpenAI(
                api_key=token,
                base_url=f"{WORKSPACE_URL}/serving-endpoints",
            )

            # Call the agent using responses.create() API
            response = client.responses.create(
                model=AGENT_ENDPOINT_NAME,
                input=[{"role": "user", "content": prompt}],
            )

            # Extract text from response.output[].content[].text
            if hasattr(response, "output") and response.output:
                texts = []
                for output in response.output:
                    if hasattr(output, "content"):
                        for item in output.content:
                            if hasattr(item, "text") and item.text:
                                texts.append(item.text)
                if texts:
                    return {"response": " ".join(texts).strip()}

            # Fallback: return raw response for debugging
            return {
                "response": str(response),
                "note": "Could not extract text from response",
            }

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "401" in error_msg:
                return {
                    "error": error_msg,
                    "message": "Authentication failed. Check that the App has serving scopes and user has Can Query permission.",
                }
            if "404" in error_msg:
                return {
                    "error": error_msg,
                    "message": f"Endpoint '{AGENT_ENDPOINT_NAME}' not found or not accessible.",
                }
            return {"error": error_msg, "message": "Failed to query the agent"}

