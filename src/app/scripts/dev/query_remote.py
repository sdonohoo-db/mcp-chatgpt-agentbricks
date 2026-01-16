#!/usr/bin/env python3
"""
Test remote MCP server deployed as a Databricks App.

This script tests the remote MCP server with user-level OAuth authentication,
calling both the health tool and user authorization tool to verify functionality.

Usage:
    python test_remote.py --host <host> --token <token> --app-url <app-url>

Example:
    python test_remote.py \\
        --host https://dbc-a1b2345c-d6e7.cloud.databricks.com \\
        --token eyJr...Dkag \\
        --app-url https://dbc-a1b2345c-d6e7.cloud.databricks.com/serving-endpoints/my-app
"""

import argparse
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OauthCredentialsStrategy, OAuthCredentialsProvider
from databricks.sdk.oauth import Token
from databricks_mcp import DatabricksMCPClient


def create_oauth_workspace_client(host: str, access_token: str) -> WorkspaceClient:
    """
    Create a WorkspaceClient using OAuth authentication.

    The Databricks SDK treats `token=` as a PAT by default. To use OAuth tokens,
    we need to use OauthCredentialsStrategy which properly identifies the auth type.
    """
    oauth_token = Token(access_token=access_token, token_type="Bearer")

    def make_oauth_provider(cfg):
        def credentials_provider():
            return {"Authorization": f"Bearer {access_token}"}

        def token_provider():
            return oauth_token

        return OAuthCredentialsProvider(
            credentials_provider=credentials_provider,
            token_provider=token_provider
        )

    strategy = OauthCredentialsStrategy(
        auth_type="oauth-u2m",
        headers_provider=make_oauth_provider
    )

    return WorkspaceClient(host=host, credentials_strategy=strategy)


def main():
    parser = argparse.ArgumentParser(
        description="Test remote MCP server deployed as Databricks App"
    )

    parser.add_argument("--host", required=True, help="Databricks workspace URL")

    parser.add_argument("--token", required=True, help="OAuth access token")

    parser.add_argument("--app-url", required=True, help="Databricks App URL (without /mcp suffix)")

    args = parser.parse_args()

    print("=" * 70)
    print("Testing Remote MCP Server - Databricks App")
    print("=" * 70)
    print(f"\nWorkspace: {args.host}")
    print(f"App URL: {args.app_url}")
    print()

    try:
        # Create WorkspaceClient with OAuth token (using proper OAuth credentials strategy)
        print("Step 1: Creating WorkspaceClient with OAuth token...")
        workspace_client = create_oauth_workspace_client(args.host, args.token)
        print("✓ WorkspaceClient created successfully (auth_type: oauth-u2m)")
        print()

        # Create MCP client
        mcp_url = f"{args.app_url}/mcp"
        print(mcp_url)
        print(f"Step 2: Connecting to MCP server at {mcp_url}...")
        mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=workspace_client)
        print("✓ MCP client connected successfully")
        print()

        # List available tools
        print("Step 3: Listing available MCP tools...")
        print("-" * 70)
        tools = mcp_client.list_tools()
        print(tools)
        print("-" * 70)
        print(f"✓ Found {len(tools) if isinstance(tools, list) else 'N/A'} tools")
        print()

        # Test arguments for tools that require parameters
        test_args = {
            "ask_agent": {"prompt": "Hello, what can you help me with?"},
        }

        for tool in tools:
            print(f"Testing tool: {tool.name}")
            print("-" * 70)
            args = test_args.get(tool.name, {})
            if args:
                print(f"Using test arguments: {args}")
            result = mcp_client.call_tool(tool.name, args)
            print(result)
            print("-" * 70)

        print("=" * 70)
        print("✓ All Tests Passed!")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Error: {e}")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
