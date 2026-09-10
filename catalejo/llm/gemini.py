"""Adaptador de Gemini: traduce una `Conversation` a la REST de Google y vuelve.

Es HTTP crudo a propósito, sin el SDK. El seam es un método y el mapeo son
sesenta líneas de JSON que se leen de una sentada; el SDK trae un árbol de
dependencias y su propio contrato async para hacer lo mismo. Cuando el mapeo
crezca (imágenes, tools, streaming) la cuenta cambia y se revisa.

Todo lo específico del proveedor vive acá adentro. El `core` no sabe que Gemini
existe, y el worker solo sabe que un modelo puede contestar o puede reventar.
"""

from __future__ import annotations

import os
from typing import Any, cast

import httpx

from catalejo.core import Conversation, Message, Role

from .model import Reply
from .reintento import ProviderError, con_reintentos

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELO = "gemini-3.8-flash"

TRANSITORIOS = frozenset({429, 500, 502, 503, 504})


CORTES_TRANSITORIOS = frozenset({"MALFORMED_FUNCTION_CALL", "OTHER"})


class GeminiError(ProviderError):
    """Lo que volvió de la API no sirve como turno.

    Hereda de `ProviderError`, que es lo que mira `con_reintentos`. Lo único que
    agrega es el nombre, y el nombre es la mitad del mensaje de error: el worker
    formatea `type(e).__name__`, así que un `Fail` dice contra qué proveedor se
    estaba hablando sin que nadie lo escriba a mano.
    """


def to_request(conv: Conversation) -> dict[str, Any]:
    """Parte la conversación en instrucción de sistema y turnos.

    Gemini quiere el sistema en un campo aparte y el resto alternando
    user/model. Dos turnos seguidos del mismo rol se juntan en uno con varias
    partes, que es un caso real y no defensa teórica: `window` mete el aviso de
    recorte como USER justo después de la pregunta, que también es USER.
    """
    sistema = "\n\n".join(m.text for m in conv if m.role is Role.SYSTEM)
    contents: list[dict[str, Any]] = []
    for m in conv:
        if m.role is Role.SYSTEM:
            continue
        role = "model" if m.role is Role.ASSISTANT else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": m.text})
        else:
            contents.append({"role": role, "parts": [{"text": m.text}]})
    if not contents:
        raise GeminiError("conversación sin turnos: el preámbulo solo no es una pregunta")
    cuerpo: dict[str, Any] = {"contents": contents}
    if sistema:
        cuerpo["systemInstruction"] = {"parts": [{"text": sistema}]}
    return cuerpo


def _solo_razonamiento(partes: list[dict[str, Any]]) -> str:
    """Si el turno vino sin texto, decirlo. Un `MALFORMED_FUNCTION_CALL` a secas no enseña.

    Este repo no declara herramientas, así que el nombre del corte no explica nada y
    manda a buscar una tool call que no existe. Lo que llega de verdad es una sola
    parte con `thoughtSignature` y `text` vacío: el modelo pensó y no dijo nada.

    Aparece por rachas según qué tenga el preámbulo, y el mensaje es lo único que
    convierte una tarde de bisección en una línea de log que dice qué pasó.
    """
    if not partes or any(p.get("text", "").strip() for p in partes if not p.get("thought")):
        return ""
    return " (el modelo devolvió solo razonamiento, sin texto)"


def from_response(payload: dict[str, Any]) -> Reply:
    """Saca el turno y el gasto, o explica por qué no hay turno.

    Tres cosas que no son obvias.

    Los resúmenes de razonamiento vienen marcados con `thought` y no son la
    respuesta. Concatenarlos rompería el contrato de terminación: un
    razonamiento que menciona código trae backticks, y el executor correría eso
    en vez de tratar el turno como respuesta final.

    Cortar por `MAX_TOKENS` es un turno roto, no un turno corto. El bloque
    cercado queda sin cerrar, `extract_code` no lo corre, el executor lo lee como
    prosa y vota DONE: el agente terminaría devolviendo media oración como si
    fuera la respuesta. Mejor que sea un `Fail` con el motivo.

    `spent` es `totalTokenCount`, que ya incluye los tokens de razonamiento. Es
    lo que cobran, así que es lo que tiene que ver el presupuesto.
    """
    candidatos = payload.get("candidates") or []
    if not candidatos:
        razon = (payload.get("promptFeedback") or {}).get("blockReason", "sin candidatos")
        raise GeminiError(f"la API no devolvió respuesta: {razon}")
    candidato = candidatos[0]
    fin = candidato.get("finishReason", "STOP")
    partes = (candidato.get("content") or {}).get("parts") or []
    if fin != "STOP":
        raise GeminiError(
            f"turno cortado por {fin}{_solo_razonamiento(partes)}",
            transitorio=fin in CORTES_TRANSITORIOS,
        )
    texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
    if not texto.strip():
        raise GeminiError("el modelo contestó vacío", transitorio=True)
    gasto = int((payload.get("usageMetadata") or {}).get("totalTokenCount", 0))
    return Reply(Message(Role.ASSISTANT, texto), spent=gasto)


class Gemini:
    """Un `Model` contra la API de Google.

    `thinking` mapea a `thinkingLevel` ("low" o "high"). Sirve para lo que hace
    el RLM: la raíz decide cuál es la próxima consulta, que es razonamiento
    barato, mientras el trabajo pesado lo hace el REPL gratis.

    `temperature` no se manda si no se pide. Los Gemini 3 vienen calibrados en
    1.0 y bajarlos a 0 buscando determinismo los empeora y los hace repetirse.
    """

    def __init__(
        self,
        model: str = MODELO,
        *,
        api_key: str | None = None,
        thinking: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 120.0,
        retries: int = 2,
        backoff: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise GeminiError("falta GEMINI_API_KEY")
        self.model = model
        self._headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        self._retries = retries
        self._backoff = backoff
        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_output_tokens is not None:
            config["maxOutputTokens"] = max_output_tokens
        if thinking is not None:
            config["thinkingConfig"] = {"thinkingLevel": thinking}
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._propio = client is None

    async def complete(self, conv: Conversation) -> Reply:
        """Un turno, reintentando lo que sea del momento.

        El reintento envuelve el pedido y el parseo, y por qué eso importa está
        contado en `reintento.py`.
        """
        cuerpo = to_request(conv)
        if self._config:
            cuerpo["generationConfig"] = self._config

        async def pedir() -> Reply:
            return from_response(await self._post(cuerpo))

        return await con_reintentos(pedir, retries=self._retries, backoff=self._backoff)

    async def _post(self, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """Un solo intento. Marca lo que se puede repetir y deja que decida arriba."""
        url = f"{BASE}/models/{self.model}:generateContent"
        try:
            r = await self._client.post(url, json=cuerpo, headers=self._headers)
        except httpx.TransportError as e:
            raise GeminiError(f"{type(e).__name__}: {e}", transitorio=True) from e
        if r.status_code != 200:
            raise GeminiError(
                f"HTTP {r.status_code}: {r.text[:300]}",
                transitorio=r.status_code in TRANSITORIOS,
            )
        datos = r.json()
        if not isinstance(datos, dict):
            raise GeminiError(f"la API devolvió {type(datos).__name__}, no un objeto")
        return cast(dict[str, Any], datos)

    async def aclose(self) -> None:
        if self._propio:
            await self._client.aclose()
