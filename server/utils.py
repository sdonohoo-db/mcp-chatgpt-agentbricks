import contextvars
import os

from databricks.sdk import WorkspaceClient

header_store = contextvars.ContextVar("header_store")


def get_workspace_client():
    return WorkspaceClient()


def get_user_token():
    """
    Get the user's OAuth token from the forwarded request headers.

    When running in Databricks Apps with OBO (On-Behalf-Of) authentication,
    the user's token is passed via the x-forwarded-access-token header.
    When running locally, returns None (local auth will be used instead).

    Returns:
        str or None: The user's OAuth token, or None if running locally.

    Raises:
        ValueError: If running in Databricks App but token is missing.
    """
    is_databricks_app = "DATABRICKS_APP_NAME" in os.environ

    if not is_databricks_app:
        return None

    headers = header_store.get({})
    token = headers.get("x-forwarded-access-token")

    if not token:
        raise ValueError(
            "Authentication token not found in request headers (x-forwarded-access-token). "
        )

    return token


def get_user_authenticated_workspace_client():
    # Check if running in a Databricks App environment
    is_databricks_app = "DATABRICKS_APP_NAME" in os.environ

    if not is_databricks_app:
        # Running locally, use default authentication
        return WorkspaceClient()

    # Running in Databricks App, require user authentication token
    token = get_user_token()
    return WorkspaceClient(token=token, auth_type="pat")
