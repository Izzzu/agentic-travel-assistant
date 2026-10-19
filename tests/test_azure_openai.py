import pytest
from openai.types.chat import ChatCompletion

from agentic.config import Settings
from agentic.core.messages import Message, ToolCall
from agentic.llm.azure_openai import AzureOpenAIClient, from_openai_response, to_openai_messages


def test_tool_call_round_trip_keeps_ids_paired() -> None:
    messages = [
        Message.system("sys"),
        Message.user("hi"),
        Message.assistant(None, [ToolCall(id="c1", name="f", arguments='{"x": 1}')]),
        Message.tool("c1", '{"ok": true}'),
    ]
    converted = to_openai_messages(messages)
    assert converted[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
        ],
    }
    assert converted[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}


def test_response_parsing() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )
    response = from_openai_response(completion)
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [ToolCall(id="c1", name="f", arguments="{}")]
    assert response.usage.total_tokens == 10


def test_missing_configuration_fails_fast() -> None:
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAIClient(Settings(azure_openai_endpoint="", azure_openai_deployment=""))
