from collections.abc import Callable

from agentic.logs.events import Event

type Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def publish(self, event: Event) -> None:
        for subscriber in self._subscribers:
            subscriber(event)
