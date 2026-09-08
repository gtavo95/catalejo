"""El vocabulario mínimo de una conversación.

Sin partes, sin tool calls, sin usage. Eso entra cuando haya un proveedor real
que lo exija, no antes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    text: str


# Una tupla, no una lista: el State es inmutable de verdad y ninguna célula
# puede quedarse con un alias de la historia del engine.
type Conversation = tuple[Message, ...]
