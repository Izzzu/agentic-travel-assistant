import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self

from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

MODEL = "scripted"


@dataclass(frozen=True)
class Call:
    """A reply that calls one tool."""

    name: str
    arguments: str = "{}"


Reply = str | Call


def _usage() -> CompletionUsage:
    return CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _completion(reply: Reply, index: int) -> ChatCompletion:
    if isinstance(reply, Call):
        message = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id=f"call-{index}",
                    type="function",
                    function=Function(name=reply.name, arguments=reply.arguments),
                )
            ],
        )
        finish = "tool_calls"
    else:
        message = ChatCompletionMessage(role="assistant", content=reply)
        finish = "stop"
    return ChatCompletion(
        id=f"reply-{index}",
        object="chat.completion",
        created=int(time.time()),
        model=MODEL,
        choices=[Choice(index=0, message=message, finish_reason=finish)],  # pyright: ignore[reportArgumentType]
        usage=_usage(),
    )


def _chunks(reply: Reply, index: int) -> list[ChatCompletionChunk]:
    def chunk(delta: ChoiceDelta, finish: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=f"reply-{index}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=MODEL,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish)],  # pyright: ignore[reportArgumentType]
        )

    if isinstance(reply, Call):
        opening = ChoiceDeltaToolCall(
            index=0,
            id=f"call-{index}",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=reply.name, arguments=""),
        )
        arguments = ChoiceDeltaToolCall(
            index=0, function=ChoiceDeltaToolCallFunction(arguments=reply.arguments)
        )
        return [
            chunk(ChoiceDelta(role="assistant", tool_calls=[opening])),
            chunk(ChoiceDelta(tool_calls=[arguments])),
            chunk(ChoiceDelta(), "tool_calls"),
        ]
    return [chunk(ChoiceDelta(role="assistant", content=reply)), chunk(ChoiceDelta(), "stop")]


class _Stream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        for item in self._chunks:
            yield item


class ScriptedOpenAI(AsyncOpenAI):
    """An OpenAI SDK client that answers with scripted replies, in order, to every request."""

    def __init__(self, replies: Sequence[Reply]) -> None:
        super().__init__(api_key="test", base_url="http://scripted.invalid/v1")
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.__dict__["chat"] = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, stream: bool = False, **request: Any) -> Any:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self._replies):
            raise AssertionError(f"the script has no reply {index + 1}")
        reply = self._replies[index]
        return _Stream(_chunks(reply, index)) if stream else _completion(reply, index)

    @property
    def unused(self) -> int:
        return len(self._replies) - len(self.requests)


def scripted_client(replies: Sequence[Reply]) -> tuple[OpenAIChatCompletionClient, ScriptedOpenAI]:
    sdk = ScriptedOpenAI(replies)
    return OpenAIChatCompletionClient(model=MODEL, async_client=sdk), sdk
