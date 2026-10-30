from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DogProfile:
    dog_id: str
    name: str
    location: str
    vaccinated: bool
    sterilized: bool
    image_key: str

    def to_item(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> "DogProfile":
        return cls(**{field: item[field] for field in cls.__dataclass_fields__})


@dataclass
class MatchResult:
    matched: bool
    score: float
    dog_id: str | None = None
    image_id: str | None = None


@dataclass
class IncomingMessage:
    message_id: str
    sender: str
    message_type: str
    text: str = ""
    media_id: str = ""


@dataclass
class OutboundMessage:
    text: str
    image_key: str | None = None
