import datetime as dt
from functools import cache
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agentic import mock_data
from agentic.core.tools import tool

AIRPORTS = {
    "zrh": "ZRH",
    "zurich": "ZRH",
    "zürich": "ZRH",
    "lis": "LIS",
    "lisbon": "LIS",
    "lisboa": "LIS",
}


class Leg(BaseModel):
    flight_no: str
    depart_time: str
    arrive_time: str
    duration: str


class Flight(BaseModel):
    id: str
    airline: str
    origin: str
    dest: str
    stops: int
    via: str | None
    outbound: Leg
    inbound: Leg
    price_per_person: float
    aircraft: str
    baggage: str
    fare_rules: str


@cache
def flights() -> tuple[Flight, ...]:
    return mock_data.load("flights.json", Flight)


def get_flight(flight_id: str) -> Flight:
    for flight in flights():
        if flight.id.lower() == flight_id.strip().lower():
            return flight
    raise LookupError(f"unknown flight {flight_id!r}; use an id from search_flights")


def _airport(name: str) -> str:
    code = AIRPORTS.get(name.strip().lower())
    if code is None:
        raise ValueError(f"no mock flights for {name!r}; only Zurich (ZRH) and Lisbon (LIS)")
    return code


@tool
def search_flights(
    origin: Annotated[str, Field(description="Departure airport code or city, e.g. ZRH")],
    dest: Annotated[str, Field(description="Arrival airport code or city, e.g. LIS")],
    depart: Annotated[dt.date, Field(description="Outbound date (YYYY-MM-DD)")],
    return_date: Annotated[dt.date, Field(description="Return date (YYYY-MM-DD)")],
    pax: Annotated[int, Field(ge=1, le=9, description="Number of travellers")],
) -> dict[str, Any]:
    """Search round-trip flights, cheapest first. Prices are in CHF."""
    route = (_airport(origin), _airport(dest))
    if return_date < depart:
        raise ValueError("return_date is before depart")
    results = [
        {
            "id": f.id,
            "airline": f.airline,
            "stops": f.stops,
            "via": f.via,
            "outbound": {"date": depart.isoformat(), **f.outbound.model_dump()},
            "inbound": {"date": return_date.isoformat(), **f.inbound.model_dump()},
            "price_per_person": f.price_per_person,
            "total_price": f.price_per_person * pax,
        }
        for f in sorted(flights(), key=lambda f: f.price_per_person)
        if (f.origin, f.dest) == route
    ]
    if not results:
        raise ValueError(f"no mock flights from {route[0]} to {route[1]}; try ZRH to LIS")
    return {"currency": "CHF", "pax": pax, "results": results}


@tool
def get_flight_details(
    flight_id: Annotated[str, Field(description="Flight id from search_flights, e.g. FL-101")],
) -> dict[str, Any]:
    """Full details of one flight option: legs, aircraft, baggage and fare rules."""
    return {"currency": "CHF", **get_flight(flight_id).model_dump()}
