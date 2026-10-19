from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from agent_framework.openai import OpenAIChatCompletionClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncOpenAI

from agentic.config import Settings
from agentic.llm.azure_openai import ENTRA_SCOPE

MAX_TOOL_ITERATIONS = 8  # model calls per agent turn, so tool results can be followed up


@asynccontextmanager
async def chat_client(settings: Settings) -> AsyncGenerator[OpenAIChatCompletionClient]:
    """A chat client on the Azure OpenAI v1 endpoint; closes the SDK client and credential on exit."""
    if not settings.azure_openai_endpoint or not settings.azure_openai_deployment:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT (see .env.example)")
    credential: DefaultAzureCredential | None = None
    if settings.azure_openai_api_key:
        api_key = settings.azure_openai_api_key.get_secret_value()
        sdk = AsyncOpenAI(api_key=api_key, base_url=_base_url(settings))
    else:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, ENTRA_SCOPE)
        sdk = AsyncOpenAI(api_key=token_provider, base_url=_base_url(settings))
    client = OpenAIChatCompletionClient(
        model=settings.azure_openai_deployment,
        async_client=sdk,
        function_invocation_configuration={"max_iterations": MAX_TOOL_ITERATIONS},
    )
    try:
        yield client
    finally:
        await sdk.close()
        if credential is not None:
            await credential.close()


def _base_url(settings: Settings) -> str:
    return f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/"
