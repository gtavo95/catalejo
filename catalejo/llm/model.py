"""El seam del modelo.

Es angosto a propósito: un adaptador traduce una conversación a la API de un
proveedor y devuelve lo que contestó. No conoce el `Log`, no vota, no decide nada.
Si la red falla, levanta excepción y el worker la convierte en un `Fail`, igual
que el executor hace con el `Environment`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from catalejo.core import Conversation, Message, Role


@dataclass(frozen=True, slots=True)
class Reply:
    """Lo que contestó el modelo, y lo que costó.

    `spent` es lo que el proveedor cobró por esta llamada. Sube por el canal
    `spent` del Log, así que el `budget` del loop ve lo que gastó el árbol
    entero, sub-modelos incluidos.
    """

    message: Message
    spent: int = 0


class Model(Protocol):
    """El puerto: lo único que el motor necesita de un proveedor.

    `worker` y `recurse` piden esto y nada más. Un `Stub` de tres líneas lo
    cumple, y por eso los tests del álgebra corren sin red.
    """

    async def complete(self, conv: Conversation) -> Reply: ...


class Provider(Protocol):
    """El puerto ancho: `Model` más el ciclo de vida y el nombre.

    Existe porque la aplicación necesita dos cosas que al motor no le importan:
    cerrar el cliente HTTP cuando termina, y decir con qué modelo contestó. Sin
    esto, `agro.py` tenía que anotarse `Gemini | OpenAI`, o sea nombrar a los dos
    adaptadores concretos para pedirles algo que el puerto no declaraba. Un
    puerto que no alcanza se paga listando implementaciones.

    Los adaptadores lo cumplen sin heredar nada. `Stub` no, a propósito: un doble
    de test no tiene cliente que cerrar, y obligarlo a inventarse un `aclose`
    sería la abstracción filtrándose hacia el otro lado.
    """

    model: str

    async def complete(self, conv: Conversation) -> Reply: ...

    async def aclose(self) -> None: ...


class Stub:
    """Un Model de mentira que contesta un guion, un mensaje por llamada.

    Guarda lo que vio en `visto`, que es como un test verifica que el contexto
    grande NUNCA llegó al prompt.
    """

    def __init__(self, *respuestas: str, spent: int = 0) -> None:
        self._respuestas = respuestas
        self._spent = spent
        self.visto: list[Conversation] = []

    async def complete(self, conv: Conversation) -> Reply:
        self.visto.append(conv)
        texto = self._respuestas[len(self.visto) - 1]
        return Reply(Message(Role.ASSISTANT, texto), spent=self._spent)
