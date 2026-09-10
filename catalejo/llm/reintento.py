"""El reintento, que es del error y no del transporte.

Vive afuera de los adaptadores porque no tiene nada de específico de un proveedor:
mira si el error se declara transitorio y repite, con la espera al doble cada vez.
Lo que sí es de cada proveedor, y por eso queda adentro de cada adaptador, es QUÉ
cuenta como transitorio: un 503 en los dos, `MALFORMED_FUNCTION_CALL` en Gemini,
un turno sin `content` en OpenAI.

Lo que envuelve es el pedido Y el parseo, y ese es el arreglo que costó dos
preguntas de veinte. Antes el reintento vivía adentro del POST, que solo ve el
código HTTP: un 200 con el turno roto salía derecho a `Fail` y el loop cortaba con
cero turnos. Quien sabe si vale la pena repetir no es el transporte, es el error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class ProviderError(RuntimeError):
    """Lo que volvió de la API no sirve como turno.

    El worker la convierte en `Fail("model", ...)` con voto QUIET, así que el loop
    corta con el motivo a la vista en vez de seguir pidiéndole texto a un proveedor
    que no está contestando.

    `transitorio` dice si reintentar el MISMO pedido tiene chance de andar. Es una
    propiedad del error y no del transporte: un 503 y un turno cortado por
    MALFORMED_FUNCTION_CALL llegan por caminos distintos y los dos se reintentan,
    un 400 y un corte por SAFETY no.
    """

    def __init__(self, mensaje: str, *, transitorio: bool = False) -> None:
        super().__init__(mensaje)
        self.transitorio = transitorio


async def con_reintentos(pedir: Callable[[], Awaitable[T]], *, retries: int, backoff: float) -> T:
    """Repite mientras el error se declare transitorio, y al rendirse dice por qué.

    El mensaje final conserva el último motivo. Sin eso, "la API no respondió"
    manda a leer los logs del proveedor para averiguar si fue cuota, red o un turno
    roto, que son tres arreglos distintos.

    Levanta la excepción de la MISMA clase que la que falló, así que quien llama
    sigue viendo `GeminiError` u `OpenAIError` y no una clase abstracta que no le
    dice contra qué estaba hablando.
    """
    espera = backoff
    ultimo: ProviderError | None = None
    for intento in range(retries + 1):
        try:
            return await pedir()
        except ProviderError as e:
            if not e.transitorio:
                raise
            ultimo = e
        if intento < retries:
            await asyncio.sleep(espera)
            espera *= 2
    if ultimo is None:
        raise ProviderError(f"retries={retries} no deja hacer ni un intento")
    raise type(ultimo)(f"la API no respondió ({retries + 1} intentos). {ultimo}")
