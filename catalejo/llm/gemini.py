"""Adaptador de Gemini: traduce una `Conversation` a la REST de Google y vuelve.

Es HTTP crudo a propósito, sin el SDK. El seam es un método y el mapeo son
sesenta líneas de JSON que se leen de una sentada; el SDK trae un árbol de
dependencias y su propio contrato async para hacer lo mismo. Cuando el mapeo
crezca (imágenes, tools, streaming) la cuenta cambia y se revisa.

Todo lo específico del proveedor vive acá adentro. El `core` no sabe que Gemini
existe, y el worker solo sabe que un modelo puede contestar o puede reventar.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, cast

import httpx

from catalejo.core import Conversation, Message, Role

from .model import Reply

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODELO = "gemini-3.8-flash"

# Fallas que suelen ser del momento: cuota por minuto, capacidad, un 500 suelto.
# Un 400 es el pedido mal armado y reintentarlo es quemar tiempo.
TRANSITORIOS = frozenset({429, 500, 502, 503, 504})


# Cortes que dependen del momento y no del prompt. MALFORMED_FUNCTION_CALL es el
# parser del servidor tropezando con su propia salida, y aparece aunque uno no
# use tools: en una corrida de 20 preguntas mató dos en la primera llamada.
# OTHER es un corte sin especificar del lado de ellos. Los demás (MAX_TOKENS,
# SAFETY, RECITATION) son deterministas: con el mismo prompt vuelve a pasar.
CORTES_TRANSITORIOS = frozenset({"MALFORMED_FUNCTION_CALL", "OTHER"})


class GeminiError(RuntimeError):
    """Lo que volvió de la API no sirve como turno.

    El worker la convierte en `Fail("model", ...)` con voto QUIET, así que el
    loop corta con el motivo a la vista en vez de seguir pidiéndole texto a un
    proveedor que no está contestando.

    `transitorio` dice si reintentar el MISMO pedido tiene chance de andar. Es
    una propiedad del error y no del transporte: un 503 y un turno cortado por
    MALFORMED_FUNCTION_CALL llegan por caminos distintos y los dos se reintentan,
    un 400 y un corte por SAFETY no.
    """

    def __init__(self, mensaje: str, *, transitorio: bool = False) -> None:
        super().__init__(mensaje)
        self.transitorio = transitorio


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
    if fin != "STOP":
        raise GeminiError(f"turno cortado por {fin}", transitorio=fin in CORTES_TRANSITORIOS)
    partes = (candidato.get("content") or {}).get("parts") or []
    texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
    if not texto.strip():
        # Con finishReason STOP y cero texto no hay nada que corregir en el
        # pedido: es la API devolviendo un turno hueco. Se pide de nuevo.
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

        El reintento envuelve el pedido Y el parseo, y eso es el arreglo. Antes
        vivía adentro de `_post`, que solo ve el código HTTP: un 200 con un turno
        cortado por MALFORMED_FUNCTION_CALL salía derecho a `Fail` y el loop
        cortaba con cero turnos. Quien sabe si vale la pena repetir no es el
        transporte, es el error.
        """
        cuerpo = to_request(conv)
        if self._config:
            cuerpo["generationConfig"] = self._config
        espera = self._backoff
        ultimo = ""
        for intento in range(self._retries + 1):
            try:
                return from_response(await self._post(cuerpo))
            except GeminiError as e:
                if not e.transitorio:
                    raise
                ultimo = str(e)
            if intento < self._retries:
                await asyncio.sleep(espera)
                espera *= 2
        raise GeminiError(f"la API no respondió ({self._retries + 1} intentos). {ultimo}")

    async def _post(self, cuerpo: dict[str, Any]) -> dict[str, Any]:
        """Un solo intento. Marca lo que se puede repetir y deja que decida arriba."""
        url = f"{BASE}/models/{self.model}:generateContent"
        try:
            r = await self._client.post(url, json=cuerpo, headers=self._headers)
        except httpx.TransportError as e:
            raise GeminiError(f"{type(e).__name__}: {e}", transitorio=True) from e
        if r.status_code != 200:
            # Un 400 es el pedido mal armado y repetirlo es quemar tiempo.
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
