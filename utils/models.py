"""Centralized model initialization"""

import os

from anthropic import APIStatusError
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
MODEL_CONFIG = {
    "model": "claude-sonnet-4-6",
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com/anthropic",
}
model = init_chat_model(
    model=MODEL_CONFIG["model"],
    model_provider=MODEL_CONFIG["provider"],
    base_url=MODEL_CONFIG["base_url"],
    api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
    max_tokens=300,
    temperature=0,
)

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )


def gateway_error_message(error: APIStatusError) -> str:
    """Describe provider failures without misidentifying unrelated errors."""
    detail = str(error).lower()
    if (
        error.status_code == 424
        and "anthropic_api_key" in detail
        and "workspace" in detail
        and "secret" in detail
        and any(term in detail for term in ("missing", "not found", "does not exist", "could not find"))
    ):
        return (
            "LangSmith gateway could not reach Anthropic: add workspace secret "
            "ANTHROPIC_API_KEY in LangSmith Settings > Secrets"
        )
    return f"LangSmith gateway provider request failed: {error}"
