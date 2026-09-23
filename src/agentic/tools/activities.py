import datetime as dt
from functools import cache
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agentic import mock_data
from agentic.core.tools import tool

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class Activity(BaseModel):
    id: str
    name: str
    city: str
    area: str
    category: str
    tags: list[str]
    price_per_person: float
    duration_hours: float
    hours: dict[str, str | None]  # weekday -> "HH:MM-HH:MM", None when closed
    notes: str = ""


@cache
def activities() -> tuple[Activity, ...]:
    return mock_data.load("activities.json", Activity)


def get_activity(activity_id: str) -> Activity:
    for activity in activities():
        if activity.id == activity_id.strip().lower():
            return activity
    raise LookupError(f"unknown activity {activity_id!r}; use an id from search_activities")


@tool
def search_activities(
    city: Annotated[str, Field(description="City, e.g. Lisbon")],
    interests: Annotated[
        list[str], Field(description="Interests to match, e.g. ['food', 'beach']; empty for all")
    ],
    date: Annotated[
        dt.date | None, Field(description="Only activities open on this date (YYYY-MM-DD)")
    ] = None,
) -> dict[str, Any]:
    """Search activities, restaurants and beaches. Prices are in CHF per person."""
    matches = [a for a in activities() if a.city.lower() == city.strip().lower()]
    if not matches:
        raise ValueError(f"no mock activities in {city!r}; only Lisbon is available")
    wanted = {i.strip().lower() for i in interests}
    weekday = WEEKDAYS[date.weekday()] if date else None
    results: list[dict[str, Any]] = []
    for a in matches:
        if wanted and wanted.isdisjoint(a.tags):
            continue
        result = a.model_dump(exclude={"hours", "city"})
        if weekday is not None:
            if a.hours.get(weekday) is None:
                continue
            result["hours_on_date"] = a.hours[weekday]
        results.append(result)
    return {"currency": "CHF", "results": results}


@tool
def get_opening_hours(
    activity_id: Annotated[str, Field(description="Activity id from search_activities")],
) -> dict[str, Any]:
    """Weekly opening hours of one activity."""
    a = get_activity(activity_id)
    return {
        "id": a.id,
        "name": a.name,
        "hours": {day: a.hours.get(day) or "closed" for day in WEEKDAYS},
        "notes": a.notes,
    }
