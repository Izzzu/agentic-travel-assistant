import datetime as dt
from functools import cache
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agentic import mock_data
from agentic.core.tools import tool


class Hotel(BaseModel):
    id: str
    name: str
    city: str
    area: str
    rating: float
    price_per_night: float
    max_guests: int
    room: str
    distance_to_center_km: float
    highlights: list[str]
    sold_out_nights: list[dt.date] = []


@cache
def hotels() -> tuple[Hotel, ...]:
    return mock_data.load("hotels.json", Hotel)


def get_hotel(hotel_id: str) -> Hotel:
    for hotel in hotels():
        if hotel.id == hotel_id.strip().lower():
            return hotel
    raise LookupError(f"unknown hotel {hotel_id!r}; use an id from search_hotels")


def _nights(checkin: dt.date, checkout: dt.date) -> list[dt.date]:
    count = (checkout - checkin).days
    if count < 1:
        raise ValueError("checkout must be after checkin")
    return [checkin + dt.timedelta(days=i) for i in range(count)]


@tool
def search_hotels(
    city: Annotated[str, Field(description="City, e.g. Lisbon")],
    checkin: Annotated[dt.date, Field(description="Check-in date (YYYY-MM-DD)")],
    checkout: Annotated[dt.date, Field(description="Check-out date (YYYY-MM-DD)")],
    guests: Annotated[int, Field(ge=1, description="Guests sharing one room or apartment")],
    max_price: Annotated[float | None, Field(description="Maximum price per night in CHF")] = None,
) -> dict[str, Any]:
    """Search accommodation, best recommendation first. Prices are in CHF per room.

    Results do not guarantee availability; confirm with check_availability.
    """
    nights = len(_nights(checkin, checkout))
    matches = [h for h in hotels() if h.city.lower() == city.strip().lower()]
    if not matches:
        raise ValueError(f"no mock hotels in {city!r}; only Lisbon is available")
    results = [
        {
            **h.model_dump(exclude={"sold_out_nights"}),
            "nights": nights,
            "total_price": h.price_per_night * nights,
        }
        for h in matches
        if h.max_guests >= guests and (max_price is None or h.price_per_night <= max_price)
    ]
    return {"currency": "CHF", "results": results}


@tool
def check_availability(
    hotel_id: Annotated[str, Field(description="Hotel id from search_hotels")],
    checkin: Annotated[dt.date, Field(description="Check-in date (YYYY-MM-DD)")],
    checkout: Annotated[dt.date, Field(description="Check-out date (YYYY-MM-DD)")],
) -> dict[str, Any]:
    """Check that a hotel still has a room for the stay. Do this before recommending it."""
    hotel = get_hotel(hotel_id)
    sold_out = [n for n in _nights(checkin, checkout) if n in hotel.sold_out_nights]
    if sold_out:
        return {
            "hotel_id": hotel.id,
            "available": False,
            "status": "sold_out",
            "sold_out_nights": [n.isoformat() for n in sold_out],
            "message": "No rooms left for these dates; pick another hotel.",
        }
    return {"hotel_id": hotel.id, "available": True, "status": "available"}
