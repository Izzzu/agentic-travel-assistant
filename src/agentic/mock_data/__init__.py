import json
from importlib.resources import files

from pydantic import BaseModel


def load[M: BaseModel](filename: str, model: type[M]) -> tuple[M, ...]:
    """Read a JSON list shipped with the package into validated models."""
    raw = json.loads(files(__name__).joinpath(filename).read_text(encoding="utf-8"))
    return tuple(model.model_validate(item) for item in raw)
