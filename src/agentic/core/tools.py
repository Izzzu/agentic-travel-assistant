import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic_core import to_json

from agentic.core.messages import ToolCall, ToolCallRecord, ToolSpec
from agentic.core.session import SessionContext
from agentic.logs.events import Event, EventType


@dataclass(frozen=True)
class ToolContext:
    """The session and the calling agent, as seen by a tool."""

    session: SessionContext
    agent: str
    run_id: str | None = None

    def emit(self, type: EventType, **payload: Any) -> Event:
        return self.session.emit(type, agent=self.agent, run_id=self.run_id, **payload)


@dataclass(frozen=True)
class Tool[**P, R]:
    fn: Callable[P, R]
    name: str
    description: str
    args_model: type[BaseModel]
    context_param: str | None = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    @property
    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)

    async def invoke(self, arguments: str, ctx: ToolContext) -> Any:
        args = self.args_model.model_validate_json(arguments or "{}")
        kwargs: dict[str, Any] = dict(args)
        if self.context_param:
            kwargs[self.context_param] = ctx
        value = cast(Callable[..., Any], self.fn)(**kwargs)
        if inspect.isawaitable(value):
            value = await value
        return value


def tool[**P, R](fn: Callable[P, R]) -> Tool[P, R]:
    """Turn a function into a tool; its docstring and type hints become the JSON schema.

    A parameter annotated `ToolContext` is injected at call time and hidden from the model.
    """
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    context_param: str | None = None
    for param in inspect.signature(fn).parameters.values():
        annotation = hints.get(param.name, Any)
        if annotation is ToolContext:
            context_param = param.name
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)
    args_model = create_model(
        f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields
    )
    return Tool(
        fn=fn,
        name=fn.__name__,
        description=inspect.cleandoc(fn.__doc__ or ""),
        args_model=args_model,
        context_param=context_param,
    )


def _error(message: str) -> str:
    return to_json({"error": message}).decode()


async def execute_tool_call(
    call: ToolCall,
    tools: Mapping[str, Tool[..., Any]],
    ctx: SessionContext,
    *,
    agent: str,
    run_id: str | None = None,
) -> ToolCallRecord:
    """Run one tool call, logging it; failures become an error result for the model."""
    ctx.emit(
        EventType.TOOL_CALL,
        agent=agent,
        run_id=run_id,
        call_id=call.id,
        tool=call.name,
        args=call.arguments,
    )
    started = time.perf_counter()
    ok = False
    try:
        selected = tools.get(call.name)
        if selected is None:
            raise LookupError(f"unknown tool {call.name!r}")
        value = await selected.invoke(call.arguments, ToolContext(ctx, agent, run_id))
        result = value if isinstance(value, str) else to_json(value).decode()
        ok = True
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()
        )
        result = _error(f"invalid arguments: {problems}")
    except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the model
        result = _error(f"{type(exc).__name__}: {exc}")
    ms = round((time.perf_counter() - started) * 1000, 1)
    ctx.emit(
        EventType.TOOL_RESULT,
        agent=agent,
        run_id=run_id,
        call_id=call.id,
        tool=call.name,
        result=result,
        ok=ok,
        ms=ms,
    )
    return ToolCallRecord(
        call_id=call.id, name=call.name, arguments=call.arguments, result=result, ok=ok, ms=ms
    )
