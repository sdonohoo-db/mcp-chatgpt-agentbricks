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

# Agent tool configuration
# DATABRICKS_HOST is automatically set by Databricks Apps runtime
# AGENT_ENDPOINT_NAME and AGENT_DESCRIPTION are set in app.yaml
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "")
AGENT_ENDPOINT_NAME = os.environ.get("AGENT_ENDPOINT_NAME", "")
AGENT_DESCRIPTION = os.environ.get("AGENT_DESCRIPTION", "Ask questions to the AI agent")


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
    def health(deep: bool = False) -> dict:
        """
        Check the health of the MCP server and optionally verify OAuth and agent connectivity.

        This diagnostic tool confirms the server is running properly. With deep=True,
        it also validates OAuth OBO authentication and agent endpoint connectivity.

        Args:
            deep: If True, performs additional checks for OAuth token validity
                  and agent endpoint connectivity. Default is False for fast health checks.

        Returns:
            dict: A dictionary containing:
                - status (str): Overall health status ("healthy", "degraded", or "unhealthy")
                - message (str): A human-readable status message
                - checks (dict, only if deep=True): Detailed results of each check

        Example response (basic):
            {
                "status": "healthy",
                "message": "MCP server is running."
            }

        Example response (deep=True, all checks pass):
            {
                "status": "healthy",
                "message": "All systems operational.",
                "checks": {
                    "server": {"status": "ok"},
                    "obo_token": {"status": "ok", "message": "Token present"},
                    "user_auth": {"status": "ok", "user": "john.doe@example.com"},
                    "agent_config": {"status": "ok", "endpoint": "my-agent"},
                    "agent_connectivity": {"status": "ok", "message": "Agent responded"}
                }
            }

        Example response (deep=True, OBO not enabled):
            {
                "status": "degraded",
                "message": "Warnings in: obo_token, user_auth",
                "checks": {
                    "server": {"status": "ok"},
                    "obo_token": {"status": "warning", "message": "No OBO token...", "hint": "Enable preview..."},
                    "user_auth": {"status": "warning", "message": "User auth unavailable..."},
                    "agent_config": {"status": "ok", "endpoint": "my-agent"},
                    "agent_connectivity": {"status": "skipped", "message": "Skipped - no OBO token"}
                }
            }
        """
        # Basic health check - always fast
        if not deep:
            return {
                "status": "healthy",
                "message": "MCP server is running.",
            }

        # Deep health check - validate all components
        checks = {}
        issues = []

        # 1. Server check (always passes if we get here)
        checks["server"] = {"status": "ok"}

        # 2. OBO token presence check
        # Note: OBO requires "Databricks Apps - On-Behalf-Of User Authorization" preview to be enabled
        token = utils.get_user_token()
        if token:
            checks["obo_token"] = {"status": "ok", "message": "Token present"}
        else:
            checks["obo_token"] = {
                "status": "warning",
                "message": "No OBO token. Expected if running locally or OBO preview not enabled.",
                "hint": "Enable 'Databricks Apps - On-Behalf-Of User Authorization' preview in workspace settings.",
            }
            issues.append("obo_token")

        # 3. User authentication check (validates token works)
        # This may fail if OBO is not enabled - treat as warning, not error
        try:
            w = utils.get_user_authenticated_workspace_client()
            user = w.current_user.me()
            checks["user_auth"] = {"status": "ok", "user": user.user_name}
        except Exception as e:
            error_msg = str(e)
            # If no token, this is expected to fail - downgrade to warning
            if not token or "token" in error_msg.lower() or "unauthorized" in error_msg.lower():
                checks["user_auth"] = {
                    "status": "warning",
                    "message": "User auth unavailable (OBO may not be enabled)",
                    "detail": error_msg,
                }
            else:
                checks["user_auth"] = {"status": "error", "message": error_msg}
            issues.append("user_auth")

        # 4. Agent configuration check
        if DATABRICKS_HOST and AGENT_ENDPOINT_NAME:
            checks["agent_config"] = {
                "status": "ok",
                "endpoint": AGENT_ENDPOINT_NAME,
                "host": DATABRICKS_HOST,
            }
        else:
            missing = []
            if not DATABRICKS_HOST:
                missing.append("DATABRICKS_HOST")
            if not AGENT_ENDPOINT_NAME:
                missing.append("AGENT_ENDPOINT_NAME")
            checks["agent_config"] = {"status": "error", "message": f"Missing: {', '.join(missing)}"}
            issues.append("agent_config")

        # 5. Agent connectivity check (only if we have token and config)
        if token and DATABRICKS_HOST and AGENT_ENDPOINT_NAME:
            try:
                host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("https://") else f"https://{DATABRICKS_HOST}"
                base_url = f"{host}/serving-endpoints"
                client = OpenAI(api_key=token, base_url=base_url)

                # Send a minimal test message
                response = client.responses.create(
                    model=AGENT_ENDPOINT_NAME,
                    input=[{"role": "user", "content": "health check"}],
                )

                # If we get here without exception, the agent is reachable
                checks["agent_connectivity"] = {"status": "ok", "message": "Agent responded"}
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg:
                    # 401 could be OBO token issue or permissions - treat as warning
                    checks["agent_connectivity"] = {
                        "status": "warning",
                        "message": "Authentication failed (401). Check OBO is enabled and user has CAN_QUERY permission.",
                    }
                elif "404" in error_msg:
                    checks["agent_connectivity"] = {"status": "error", "message": f"Endpoint '{AGENT_ENDPOINT_NAME}' not found (404)"}
                else:
                    checks["agent_connectivity"] = {"status": "error", "message": error_msg}
                issues.append("agent_connectivity")
        elif not token:
            checks["agent_connectivity"] = {
                "status": "skipped",
                "message": "Skipped - no OBO token available",
            }
        else:
            checks["agent_connectivity"] = {"status": "skipped", "message": "Skipped - missing agent config"}

        # Determine overall status
        error_checks = [c for c in issues if checks.get(c, {}).get("status") == "error"]
        if error_checks:
            status = "unhealthy"
            message = f"Errors in: {', '.join(error_checks)}"
        elif issues:
            status = "degraded"
            message = f"Warnings in: {', '.join(issues)}"
        else:
            status = "healthy"
            message = "All systems operational."

        return {
            "status": status,
            "message": message,
            "checks": checks,
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

    # Define ask_agent with dynamic docstring from AGENT_DESCRIPTION
    def ask_agent(prompt: str) -> dict:
        """Placeholder docstring - replaced dynamically."""
        try:
            # Get the user's OBO token
            token = utils.get_user_token()

            if token is None:
                return {
                    "error": "No OBO token available",
                    "message": "This tool requires OBO authentication. Running locally without token.",
                }

            # Validate configuration
            if not DATABRICKS_HOST:
                return {
                    "error": "DATABRICKS_HOST not configured",
                    "message": "The DATABRICKS_HOST environment variable is not set. This should be automatic in Databricks Apps.",
                }
            if not AGENT_ENDPOINT_NAME:
                return {
                    "error": "AGENT_ENDPOINT_NAME not configured",
                    "message": "The AGENT_ENDPOINT_NAME environment variable is not set.",
                }

            # Create OpenAI client pointing to Databricks serving endpoints
            # Ensure DATABRICKS_HOST has https:// prefix
            host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("https://") else f"https://{DATABRICKS_HOST}"
            base_url = f"{host}/serving-endpoints"
            client = OpenAI(
                api_key=token,
                base_url=base_url,
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
            # Normalize host for debug output
            debug_host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("https://") else f"https://{DATABRICKS_HOST}"
            return {
                "error": error_msg,
                "message": "Failed to query the agent",
                "debug": {
                    "base_url": f"{debug_host}/serving-endpoints",
                    "endpoint": AGENT_ENDPOINT_NAME,
                },
            }

    # Set the docstring dynamically from AGENT_DESCRIPTION environment variable
    ask_agent.__doc__ = f"""{AGENT_DESCRIPTION}

    Args:
        prompt: The question or message to send to the agent.

    Returns:
        dict: The agent's response or an error message.
    """

    # Register the tool with the MCP server
    mcp_server.tool()(ask_agent)

