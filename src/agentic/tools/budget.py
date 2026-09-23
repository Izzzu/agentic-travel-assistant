from typing import Annotated, Any

from pydantic import BaseModel, Field

from agentic.core.tools import tool
from agentic.tools.activities import activities
from agentic.tools.flight import flights
from agentic.tools.hotel import hotels


class CostItem(BaseModel):
    label: str = Field(description="What the cost is, e.g. 'Hotel baixa-boutique'")
    unit_price: float = Field(ge=0, description="Price per unit in CHF")
    quantity: int = Field(default=1, ge=1, description="Units, e.g. travellers or nights")
    ref_id: str | None = Field(
        default=None, description="Id of the flight, hotel or activity this cost is for"
    )

    @property
    def subtotal(self) -> float:
        return round(self.unit_price * self.quantity, 2)


def _total(items: list[CostItem]) -> float:
    return round(sum(i.subtotal for i in items), 2)


def _alternatives(item: CostItem) -> list[tuple[str, str, float, str]]:
    """Cheaper catalog options of the same kind as `item`: (id, name, unit price, note)."""
    ref = (item.ref_id or "").strip().lower()
    if any(f.id.lower() == ref for f in flights()):
        return [
            (f.id, f"{f.airline} {f.id}", f.price_per_person, f"arrives {f.outbound.arrive_time}")
            for f in flights()
            if f.id.lower() != ref and f.price_per_person < item.unit_price
        ]
    if any(h.id == ref for h in hotels()):
        return [
            (
                h.id,
                h.name,
                h.price_per_night,
                f"{h.area}, {h.distance_to_center_km} km from center; check availability first",
            )
            for h in hotels()
            if h.id != ref and h.price_per_night < item.unit_price
        ]
    current = next((a for a in activities() if a.id == ref), None)
    if current is None:
        return []
    return [
        (a.id, a.name, a.price_per_person, f"{a.category} in {a.area}")
        for a in activities()
        if a.id != ref and a.price_per_person < item.unit_price and current.tags[0] in a.tags
    ]


@tool
def calculate_total(items: list[CostItem]) -> dict[str, Any]:
    """Add up plan costs in CHF. Always use this instead of doing the math yourself."""
    return {
        "currency": "CHF",
        "items": [{**i.model_dump(), "subtotal": i.subtotal} for i in items],
        "total": _total(items),
    }


@tool
def check_budget(
    total: Annotated[float, Field(ge=0, description="Plan total in CHF")],
    limit: Annotated[float, Field(gt=0, description="Budget limit in CHF")],
) -> dict[str, Any]:
    """Compare a plan total with the budget."""
    difference = round(limit - total, 2)
    return {
        "currency": "CHF",
        "total": total,
        "limit": limit,
        "within_budget": total <= limit,
        "over_by": max(0.0, -difference),
        "remaining": max(0.0, difference),
        "used_percent": round(total / limit * 100, 1),
    }


@tool
def suggest_savings(
    items: list[CostItem],
    target: Annotated[float, Field(gt=0, description="Total the plan must fit, in CHF")],
) -> dict[str, Any]:
    """Cheaper alternatives for plan items, biggest saving first, plus a set that closes the gap.

    Give each item its ref_id so alternatives can be found.
    """
    total = _total(items)
    needed = round(max(0.0, total - target), 2)
    options: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(items):
        for ref_id, name, price, note in _alternatives(item):
            saving = round((item.unit_price - price) * item.quantity, 2)
            options.append(
                (
                    index,
                    {
                        "replace": item.label,
                        "with": name,
                        "ref_id": ref_id,
                        "unit_price": price,
                        "saving": saving,
                        "note": note,
                    },
                )
            )
    options.sort(key=lambda o: o[1]["saving"], reverse=True)
    recommended: list[dict[str, Any]] = []
    swapped: set[int] = set()
    saved = 0.0
    for index, option in options:
        if saved >= needed:
            break
        if index not in swapped:
            recommended.append(option)
            swapped.add(index)
            saved += option["saving"]
    return {
        "currency": "CHF",
        "total": total,
        "target": target,
        "needed": needed,
        "suggestions": [option for _, option in options],
        "recommended": recommended,
        "total_after_recommended": round(total - saved, 2),
        "reaches_target": saved >= needed,
    }
