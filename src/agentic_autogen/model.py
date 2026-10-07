from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from agentic_autogen.config import AutogenSettings

ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"

# Deployment names are not in AutoGen's model table, so the capabilities are stated here.
MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    structured_output=True,
    family="unknown",
)


@asynccontextmanager
async def model_client(
    settings: AutogenSettings, *, parallel_tool_calls: bool | None = None
) -> AsyncGenerator[AzureOpenAIChatCompletionClient]:
    """An Azure OpenAI client for AutoGen; closes the client and its credential on exit."""
    if not settings.azure_openai_endpoint or not settings.azure_openai_deployment:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT (see .env.example)")
    options: dict[str, Any] = {}
    if parallel_tool_calls is not None:
        options["parallel_tool_calls"] = parallel_tool_calls
    credential: DefaultAzureCredential | None = None
    if settings.azure_openai_api_key:
        options["api_key"] = settings.azure_openai_api_key.get_secret_value()
    else:
        credential = DefaultAzureCredential()
        options["azure_ad_token_provider"] = get_bearer_token_provider(credential, ENTRA_SCOPE)
    client = AzureOpenAIChatCompletionClient(
        azure_deployment=settings.azure_openai_deployment,
        model=settings.model,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        model_info=MODEL_INFO,
        **options,
    )
    try:
        yield client
    finally:
        await client.close()
        if credential is not None:
            await credential.close()
