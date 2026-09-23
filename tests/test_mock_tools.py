import json
from typing import Any

import pytest

from agentic.core.agent import parse_open_questions
from agentic.core.messages import ToolCall
from agentic.core.session import SessionContext
from agentic.core.tools import Tool, execute_tool_call
from agentic.io.user_io import ScriptedIO
from agentic.logs.bus import EventBus
from agentic.logs.events import Event, EventType
from agentic.tools.activities import get_opening_hours, search_activities
from agentic.tools.budget import calculate_total, check_budget, suggest_savings
from agentic.tools.flight import get_flight_details, search_flights
from agentic.tools.hotel import check_availability, search_hotels
from agentic.tools.shared import ask_user

DEMO_WEEKEND = {"checkin": "2026-10-09", "checkout": "2026-10-11"}  # Fri-Sun; Casa Alfama is full
QUIET_WEEKEND = {"checkin": "2026-10-16", "checkout": "2026-10-18"}  # Casa Alfama is free
TRIP = DEMO_WEEKEND
BUDGET = 1500

# The plan a naive pipeline builds once the best-value hotel is gone.
NAIVE_PLAN = [
    {"label": "Flights", "unit_price": 210, "quantity": 3, "ref_id": "FL-102"},
    {"label": "Hotel", "unit_price": 220, "quantity": 2, "ref_id": "baixa-boutique"},
    {"label": "Food tour", "unit_price": 85, "quantity": 3, "ref_id": "alfama-food-tour"},
    {"label": "Coast tour", "unit_price": 95, "quantity": 3, "ref_id": "cascais-coast-tour"},
    {"label": "Dinner", "unit_price": 30, "quantity": 3, "ref_id": "time-out-market"},
]


class Session:
    def __init__(self, *, answers: tuple[str, ...] = ()) -> None:
        self.io = ScriptedIO(answers)
        self.events: list[Event] = []
        bus = EventBus()
        bus.subscribe(self.events.append)
        self.ctx = SessionContext(pattern="test", bus=bus, io=self.io)

    async def call(self, fn: Tool[..., Any], **args: Any) -> Any:
        """Invoke a tool the way the agent loop does: JSON in, JSON (or plain text) out."""
        call = ToolCall(id="t1", name=fn.name, arguments=json.dumps(args))
        record = await execute_tool_call(call, {fn.name: fn}, self.ctx, agent="tester")
        assert record.ok, record.result
        try:
            return json.loads(record.result)
        except json.JSONDecodeError:
            return record.result

    async def fail(self, fn: Tool[..., Any], **args: Any) -> str:
        call = ToolCall(id="t1", name=fn.name, arguments=json.dumps(args))
        record = await execute_tool_call(call, {fn.name: fn}, self.ctx, agent="tester")
        assert not record.ok
        return json.loads(record.result)["error"]

    def of(self, type: EventType) -> list[Event]:
        return [e for e in self.events if e.type is type]


# --- ask_user ---


async def test_ask_user_asks_through_io_and_logs_both_sides() -> None:
    s = Session(answers=("Yes, a shared room is fine",))
    answer = await s.call(ask_user, question="Is a shared room OK?")
    assert answer == "Yes, a shared room is fine"
    assert s.io.asked == [("tester", "Is a shared room OK?")]
    asked, replied = s.of(EventType.ASK_USER), s.of(EventType.USER_REPLY)
    assert asked[0].agent == replied[0].agent == "tester"
    assert replied[0].payload["answer"] == answer


# --- flights ---


async def test_search_flights_cheapest_first_with_totals() -> None:
    result = await Session().call(
        search_flights,
        origin="Zurich",
        dest="lis",
        depart="2026-10-09",
        return_date="2026-10-11",
        pax=3,
    )
    options = result["results"]
    prices = [o["price_per_person"] for o in options]
    assert prices == sorted(prices) and len(options) == 6
    assert all(o["total_price"] == o["price_per_person"] * 3 for o in options)
    assert options[0]["outbound"] == {
        "date": "2026-10-09",
        "flight_no": "TP937",
        "depart_time": "20:15",
        "arrive_time": "22:00",  # the cheap flight lands late
        "duration": "2h45",
    }


async def test_search_flights_rejects_bad_input() -> None:
    s = Session()
    base = {"depart": "2026-10-09", "return_date": "2026-10-11", "pax": 3}
    assert "only Zurich" in await s.fail(search_flights, origin="Paris", dest="LIS", **base)
    assert "no mock flights from LIS" in await s.fail(
        search_flights, origin="LIS", dest="ZRH", **base
    )
    bad_dates = {**base, "return_date": "2026-10-01"}
    assert "before depart" in await s.fail(search_flights, origin="ZRH", dest="LIS", **bad_dates)


async def test_get_flight_details() -> None:
    s = Session()
    details = await s.call(get_flight_details, flight_id="fl-103")
    assert (details["stops"], details["via"]) == (1, "MAD")
    assert "unknown flight" in await s.fail(get_flight_details, flight_id="XX-1")


# --- hotels and availability ---


async def test_search_hotels_lists_best_value_first_without_availability() -> None:
    result = await Session().call(search_hotels, city="Lisbon", guests=3, **TRIP)
    hotels = result["results"]
    first = hotels[0]
    assert first["id"] == "casa-alfama"
    assert first["price_per_night"] == min(h["price_per_night"] for h in hotels)
    assert (first["nights"], first["total_price"]) == (2, 240)
    assert all("sold_out_nights" not in h for h in hotels)


async def test_search_hotels_filters() -> None:
    s = Session()
    cheap = await s.call(search_hotels, city="lisbon", guests=3, max_price=160, **TRIP)
    assert [h["id"] for h in cheap["results"]] == ["casa-alfama", "belem-riverside"]
    big = await s.call(search_hotels, city="Lisbon", guests=4, **TRIP)
    assert {h["id"] for h in big["results"]} == {
        "baixa-boutique",
        "belem-riverside",
        "principe-real-house",
        "cascais-beach-apartment",
    }
    same_day = {"checkin": "2026-10-09", "checkout": "2026-10-09"}
    assert "checkout" in await s.fail(search_hotels, city="Lisbon", guests=3, **same_day)
    assert "only Lisbon" in await s.fail(search_hotels, city="Porto", guests=3, **TRIP)


async def test_best_value_hotel_is_sold_out_on_the_demo_weekend_only() -> None:
    s = Session()
    full = await s.call(check_availability, hotel_id="casa-alfama", **DEMO_WEEKEND)
    assert (full["available"], full["status"]) == (False, "sold_out")
    assert full["sold_out_nights"] == ["2026-10-09", "2026-10-10"]
    free = await s.call(check_availability, hotel_id="casa-alfama", **QUIET_WEEKEND)
    assert free["available"] is True
    # the next recommendation is free on the demo weekend
    assert (await s.call(check_availability, hotel_id="baixa-boutique", **DEMO_WEEKEND))[
        "available"
    ] is True


async def test_partly_sold_out_stay() -> None:
    s = Session()
    stay = {"checkin": "2026-10-22", "checkout": "2026-10-24"}
    result = await s.call(check_availability, hotel_id="graca-guesthouse", **stay)
    assert result["sold_out_nights"] == ["2026-10-23"]
    assert "checkout" in await s.fail(
        check_availability, hotel_id="graca-guesthouse", checkin="2026-10-24", checkout="2026-10-22"
    )


# --- activities ---


async def test_search_activities_by_interest() -> None:
    result = await Session().call(search_activities, city="Lisbon", interests=["Beach"])
    ids = [a["id"] for a in result["results"]]
    assert ids == ["cascais-coast-tour", "cascais-beach-day", "caparica-surf", "carcavelos-beach"]
    everything = await Session().call(search_activities, city="Lisbon", interests=[])
    assert len(everything["results"]) == 13


async def test_search_activities_by_date_uses_opening_hours() -> None:
    monday = await Session().call(
        search_activities, city="Lisbon", interests=["food"], date="2026-10-12"
    )
    ids = {a["id"] for a in monday["results"]}
    assert "alfama-food-tour" not in ids and "ramiro-seafood" not in ids
    friday = await Session().call(
        search_activities, city="Lisbon", interests=["food"], date="2026-10-09"
    )
    tour = next(a for a in friday["results"] if a["id"] == "alfama-food-tour")
    assert tour["hours_on_date"] == "10:00-13:30"


async def test_get_opening_hours() -> None:
    s = Session()
    hours = await s.call(get_opening_hours, activity_id="cascais-coast-tour")
    assert hours["hours"]["fri"] == "closed"
    assert hours["hours"]["sat"] == "09:00-17:00"
    assert "unknown activity" in await s.fail(get_opening_hours, activity_id="nope")


# --- budget math ---


async def test_calculate_total() -> None:
    result = await Session().call(calculate_total, items=NAIVE_PLAN)
    assert [i["subtotal"] for i in result["items"]] == [630, 440, 255, 285, 90]
    assert result["total"] == 1700


async def test_check_budget() -> None:
    s = Session()
    over = await s.call(check_budget, total=1700, limit=BUDGET)
    assert (over["within_budget"], over["over_by"], over["remaining"]) == (False, 200, 0)
    under = await s.call(check_budget, total=1290, limit=BUDGET)
    assert (under["within_budget"], under["over_by"], under["remaining"]) == (True, 0, 210)
    assert "limit" in await s.fail(check_budget, total=100, limit=0)


async def test_scenario_tuning() -> None:
    """With the best-value hotel the naive plan fits; without it, it is CHF 200 over; a fix exists."""
    s = Session()
    with_alfama = [
        *NAIVE_PLAN[:1],
        {"label": "Hotel", "unit_price": 120, "quantity": 2, "ref_id": "casa-alfama"},
        *NAIVE_PLAN[2:],
    ]
    assert (await s.call(calculate_total, items=with_alfama))["total"] == BUDGET
    assert (await s.call(calculate_total, items=NAIVE_PLAN))["total"] == BUDGET + 200
    valid = [
        *NAIVE_PLAN[:1],
        {"label": "Hotel", "unit_price": 150, "quantity": 2, "ref_id": "belem-riverside"},
        NAIVE_PLAN[2],
        {"label": "Beach day", "unit_price": 5, "quantity": 3, "ref_id": "cascais-beach-day"},
        NAIVE_PLAN[4],
    ]
    assert (await s.call(calculate_total, items=valid))["total"] == 1290


async def test_suggest_savings_closes_the_gap() -> None:
    result = await Session().call(suggest_savings, items=NAIVE_PLAN, target=BUDGET)
    assert (result["total"], result["needed"]) == (1700, 200)
    assert result["reaches_target"] is True
    assert result["total_after_recommended"] <= BUDGET
    best = result["recommended"][0]
    assert (best["replace"], best["ref_id"], best["saving"]) == (
        "Coast tour",
        "carcavelos-beach",
        273,
    )
    hotel_swaps = {o["ref_id"] for o in result["suggestions"] if o["replace"] == "Hotel"}
    assert hotel_swaps == {
        "casa-alfama",
        "cais-loft",
        "belem-riverside",
        "graca-guesthouse",
        "cascais-beach-apartment",
    }
    tour_swaps = {o["ref_id"] for o in result["suggestions"] if o["replace"] == "Coast tour"}
    assert "tram-28" not in tour_swaps  # a beach outing is only swapped for another beach


async def test_suggest_savings_nothing_needed() -> None:
    result = await Session().call(suggest_savings, items=NAIVE_PLAN, target=2000)
    assert (result["needed"], result["recommended"], result["reaches_target"]) == (0, [], True)


# --- open questions ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Hotel: Baixa.\n\nOpen questions:\n- Shared room OK?\n- Breakfast?",
            ["Shared room OK?", "Breakfast?"],
        ),
        ("Plan.\n\n**Open questions:**\n\n1. Late check-in?\n\nNotes follow.", ["Late check-in?"]),
        ("## Open questions\n* Beach or pool?\nThanks", ["Beach or pool?"]),
        ("Open questions: Is a 22:00 arrival OK?", ["Is a 22:00 arrival OK?"]),
        ("Open questions: none", []),
        ("No questions here. Open questions about hotels are fine.", []),
    ],
)
def test_parse_open_questions(text: str, expected: list[str]) -> None:
    assert parse_open_questions(text) == expected
