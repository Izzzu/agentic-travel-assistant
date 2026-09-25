from collections.abc import Sequence

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel

from agentic.config import Settings
from agentic.core.messages import FinishReason, LLMResponse, Message, ToolCall, ToolSpec, Usage

ENTRA_SCOPE = "https://cognitiveservices.azure.com/.default"


def to_openai_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    converted: list[ChatCompletionMessageParam] = []
    for m in messages:
        match m.role:
            case "system":
                converted.append({"role": "system", "content": m.content or ""})
            case "user":
                converted.append({"role": "user", "content": m.content or ""})
            case "assistant":
                assistant: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": m.content,
                }
                if m.tool_calls:
                    calls: list[ChatCompletionMessageFunctionToolCallParam] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in m.tool_calls
                    ]
                    assistant["tool_calls"] = calls
                converted.append(assistant)
            case "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "",
                        "content": m.content or "",
                    }
                )
    return converted


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[ChatCompletionFunctionToolParam]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def from_openai_response(completion: ChatCompletion) -> LLMResponse:
    choice = completion.choices[0]
    message = choice.message
    tool_calls = [
        ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments)
        for c in message.tool_calls or []
        if c.type == "function"
    ]
    finish: FinishReason = (
        choice.finish_reason
        if choice.finish_reason in ("stop", "length", "tool_calls", "content_filter")
        else "other"
    )
    usage = completion.usage
    return LLMResponse(
        model=completion.model,
        content=message.content if message.content is not None else message.refusal,
        tool_calls=tool_calls,
        finish_reason=finish,
        usage=Usage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        ),
    )


class AzureOpenAIClient:
    """Chat Completions against the Azure OpenAI v1 endpoint (no api-version needed).

    Uses the API key when set, otherwise Microsoft Entra ID via DefaultAzureCredential.
    Transient failures are retried with exponential backoff by the OpenAI SDK.
    """

    def __init__(self, settings: Settings, *, max_retries: int = 3, timeout: float = 60.0) -> None:
        if not settings.azure_openai_endpoint or not settings.azure_openai_deployment:
            raise ValueError(
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT (see .env.example)"
            )
        self._credential: DefaultAzureCredential | None = None
        if settings.azure_openai_api_key:
            api_key = settings.azure_openai_api_key.get_secret_value()
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/",
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            self._credential = DefaultAzureCredential()
            self._client = AsyncOpenAI(
                api_key=get_bearer_token_provider(self._credential, ENTRA_SCOPE),
                base_url=f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/",
                max_retries=max_retries,
                timeout=timeout,
            )
        self._deployment = settings.azure_openai_deployment

    @property
    def model(self) -> str:
        return self._deployment

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        format_param: ResponseFormatJSONSchema | None = None
        if response_format is not None:
            format_param = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }
        completion = await self._client.chat.completions.create(
            model=self._deployment,
            messages=to_openai_messages(messages),
            tools=to_openai_tools(tools) if tools else omit,
            response_format=format_param if format_param else omit,
        )
        return from_openai_response(completion)

    async def aclose(self) -> None:
        await self._client.close()
        if self._credential is not None:
            await self._credential.close()
