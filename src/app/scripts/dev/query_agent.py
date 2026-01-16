#!/usr/bin/env python3
"""
Test Agent Bricks endpoint directly.

This script tests a Databricks Agent Bricks endpoint directly using the OpenAI
client, bypassing the MCP server. Useful for verifying the agent endpoint is
working before testing through the MCP server.

Usage:
    python query_agent.py --host <host> --token <token> --endpoint <endpoint> --prompt <prompt>

Example:
    python query_agent.py \
        --host https://adb-1234567890.12.azuredatabricks.net \
        --token eyJr...Dkag \
        --endpoint my-agent-endpoint \
        --prompt "What can you help me with?"
"""

import argparse
import sys

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(
        description="Test Agent Bricks endpoint directly"
    )

    parser.add_argument("--host", required=True, help="Databricks workspace URL")

    parser.add_argument("--token", required=True, help="Databricks access token")

    parser.add_argument("--endpoint", required=True, help="Agent endpoint name")

    parser.add_argument("--prompt", required=True, help="Prompt to send to the agent")

    args = parser.parse_args()

    print("=" * 70)
    print("Testing Agent Bricks Endpoint")
    print("=" * 70)
    print(f"\nWorkspace: {args.host}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Prompt: {args.prompt}")
    print()

    try:
        # Create OpenAI client pointing to Databricks serving endpoints
        print("Step 1: Creating OpenAI client...")
        client = OpenAI(
            api_key=args.token,
            base_url=f"{args.host}/serving-endpoints"
        )
        print("✓ OpenAI client created successfully")
        print()

        # Call the agent endpoint
        print("Step 2: Calling agent endpoint...")
        print("-" * 70)
        response = client.responses.create(
            model=args.endpoint,
            input=[
                {
                    "role": "user",
                    "content": args.prompt
                }
            ]
        )

        # Extract text from response
        text = " ".join(
            getattr(content, "text", "")
            for output in response.output
            for content in getattr(output, "content", [])
        )
        print(text)
        print("-" * 70)
        print()

        print("=" * 70)
        print("✓ Query completed successfully!")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Error: {e}")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
