"""Adaptador de OpenAI: la misma `Conversation` contra /chat/completions.

Habla Chat Completions y no la Responses API, y `base_url` es un parámetro, así
que este archivo sirve para todo lo que copia ese contrato: Groq, Together,
OpenRouter, vLLM, Ollama. La Responses API traería los ítems de razonamiento
cifrados para reenviarlos entre turnos, y acá no sirven: este repo no declara
herramientas y cada turno es prosa o un bloque de código, sin estado del lado del
proveedor.

El mapeo es la mitad que el de Gemini, y no por mérito nuestro. El formato ya se
parece al `core`: el sistema es un mensaje más en la lista, `Role` es un StrEnum
cuyos valores son literalmente los que espera la API, y dos turnos seguidos del
mismo rol se aceptan sin juntarlos, que en Gemini hubo que resolver a mano.

Existe por lo mismo que existe el `retries=4` de `agro.py`. Un proveedor que
devuelve turnos vacíos por rachas es un piso de fiabilidad que no se sube desde
acá, y tener el segundo adaptador convierte "esperemos que hoy conteste" en
cambiar una línea. Que las dos clases tomen los mismos parámetros no es
casualidad, es el requisito para que ese cambio sea de una línea.
"""

from __future__ import annotations

import os
from typing import Any, cast

import httpx

from catalejo.core import Conversation, Message, Role

from .model import Reply
from .reintento import ProviderError, con_reintentos

BASE = "https://api.openai.com/v1"
MODELO = "gpt-5.1"

TRANSITORIOS = frozenset({408, 429, 500, 502, 503, 504})

CORTES_TRANSITORIOS = frozenset({"tool_calls", "function_call"})


class OpenAIError(ProviderError):
    """Lo que volvió de la API no sirve como turno.

    Hereda de `ProviderError`, que es lo que mira `con_reintentos`. Lo único que
    agrega es el nombre, y el nombre es la mitad del mensaje de error: el worker
    formatea `type(e).__name__`, así que un `Fail` dice contra qué proveedor se
    estaba hablando sin que nadie lo escriba a mano.
    """


def to_messages(conv: Conversation) -> list[dict[str, str]]:
    """La conversación tal cual, que acá es una lista de {role, content}.

    `Role` es un StrEnum con los mismos valores que usa la API, así que no hay
    traducción que hacer. Es una coincidencia afortunada y conviene decirla: si
    mañana el core gana un rol que la API no tiene, esto deja de ser un `.value`
    y pasa a ser un diccionario.

    Una conversación con puras instrucciones de sistema no es una pregunta. La
    API contestaría igual, inventando de qué hablar, y eso es peor que el error.
    """
    if not any(m.role is not Role.SYSTEM for m in conv):
        raise OpenAIError("conversación sin turnos: el preámbulo solo no es una pregunta")
    return [{"role": m.role.value, "content": m.text} for m in conv]


def from_completion(payload: dict[str, Any]) -> Reply:
    """Saca el turno y el gasto, o explica por qué no hay turno.

    Cuatro cosas que no son obvias.

    Un `refusal` viene en su propio campo con el `content` vacío. Leerlo es la
    diferencia entre "el modelo contestó vacío" y saber que se negó, que son dos
    problemas distintos y uno de los dos no se arregla reintentando.

    Cortar por `length` es un turno roto, no un turno corto. El bloque cercado
    queda sin cerrar, `extract_code` no lo corre, el executor lo lee como prosa y
    vota DONE: el agente devolvería media oración como respuesta final. No se
    reintenta, porque el mismo prompt vuelve a cortar en el mismo lugar.

    Un turno con `content` vacío y sin refusal es el mismo agujero que en Gemini.
    Los modelos de razonamiento pueden gastar todo el presupuesto de salida
    pensando y no escribir nada, y llega como un 200 perfectamente formado. Se
    reintenta porque el mismo pedido repetido a veces sale: contra Gemini,
    doce llamadas idénticas daban seis turnos buenos y seis vacíos.

    `spent` es `usage.total_tokens`, que ya incluye los tokens de razonamiento.
    Es lo que cobran, así que es lo que tiene que ver el presupuesto.
    """
    error = payload.get("error")
    if error:
        detalle = error.get("message", error) if isinstance(error, dict) else error
        raise OpenAIError(f"la API devolvió un error: {detalle}")
    opciones = payload.get("choices") or []
    if not opciones:
        raise OpenAIError("la API no devolvió respuesta: sin choices")
    opcion = opciones[0]
    mensaje = opcion.get("message") or {}
    negativa = (mensaje.get("refusal") or "").strip()
    if negativa:
        raise OpenAIError(f"el modelo se negó: {negativa}")
    fin = opcion.get("finish_reason") or "stop"
    if fin != "stop":
        raise OpenAIError(
            f"turno cortado por {fin}",
            transitorio=fin in CORTES_TRANSITORIOS,
        )
    texto = mensaje.get("content")
    if not isinstance(texto, str) or not texto.strip():
        raise OpenAIError(
            "el modelo contestó vacío (turno sin content: suele ser todo el "
            "presupuesto de salida gastado en razonamiento)",
            transitorio=True,
        )
    gasto = int((payload.get("usage") or {}).get("total_tokens", 0))
    return Reply(Message(Role.ASSISTANT, texto), spent=gasto)


class OpenAI:
    """Un `Model` contra /chat/completions.

    Los parámetros son los mismos que los de `Gemini`, a propósito. `thinking`
    mapea a `reasoning_effort` ("minimal", "low", "medium", "high") en vez de a
    `thinkingLevel`, y el que llama no se entera: cambiar de proveedor es cambiar
    la clase.

    `temperature` no se manda si no se pide, y acá no es una preferencia como en
    Gemini: los modelos de razonamiento rechazan cualquier valor que no sea 1.

    `max_output_tokens` va como `max_completion_tokens`. El viejo `max_tokens`
    sigue documentado pero los modelos de razonamiento lo rechazan, y el error
    que devuelven no dice cuál es el nombre nuevo.

    `base_url` es lo que hace que esto valga para más de un proveedor. La key se
    exige solo contra api.openai.com, porque un endpoint local no pide ninguna y
    obligar a inventarse una sería pedir un trámite por nada.
    """

    def __init__(
        self,
        model: str = MODELO,
        *,
        api_key: str | None = None,
        base_url: str = BASE,
        thinking: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        retries: int = 2,
        backoff: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key and base_url == BASE:
            raise OpenAIError("falta OPENAI_API_KEY")
        self.model = model
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"
        self._retries = retries
        self._backoff = backoff
        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_output_tokens is not None:
            config["max_completion_tokens"] = max_output_tokens
        if thinking is not None:
            config["reasoning_effort"] = thinking
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._propio = client is None

    async def complete(self, conv: Conversation) -> Reply:
        """Un turno, reintentando lo que sea del momento.

        El reintento envuelve el pedido y el parseo, y por qué eso importa está
        contado en `reintento.py`.
        """
        cuerpo: dict[str, Any] = {
            "model": self.model,
            "messages": to_messages(conv),
            **self._config,
        }

        async def pedir() -> Reply:
            return from_completion(await self._post(cuerpo))

        return await con_reintentos(pedir, retries=self._retries, backoff=self._backoff)

    async def _post(self, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """Un solo intento. Marca lo que se puede repetir y deja que decida arriba."""
        try:
            r = await self._client.post(self._url, json=cuerpo, headers=self._headers)
        except httpx.TransportError as e:
            raise OpenAIError(f"{type(e).__name__}: {e}", transitorio=True) from e
        if r.status_code != 200:
            raise OpenAIError(
                f"HTTP {r.status_code}: {r.text[:300]}",
                transitorio=r.status_code in TRANSITORIOS,
            )
        datos = r.json()
        if not isinstance(datos, dict):
            raise OpenAIError(f"la API devolvió {type(datos).__name__}, no un objeto")
        return cast(dict[str, Any], datos)

    async def aclose(self) -> None:
        if self._propio:
            await self._client.aclose()
