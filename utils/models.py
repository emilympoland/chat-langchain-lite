"""Centralized model initialization"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# The frontend's Gateway pane reads the same configuration as the agent.
def get_model_config(model_id: str | None = None) -> dict:
    return {
        "model": model_id or os.getenv("CHAT_LANGCHAIN_LITE_MODEL") or "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "base_url": "https://gateway.smith.langchain.com/anthropic",
    }


def get_model(model_id: str | None = None):
    config = get_model_config(model_id)
    return init_chat_model(
        model=config["model"],
        model_provider=config["provider"],
        base_url=config["base_url"],
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