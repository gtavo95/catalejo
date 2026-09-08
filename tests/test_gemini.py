import json
import os
from typing import Any

import httpx
import pytest

from catalejo.core import ZERO, Fail, Log, Message, Role, loop, then
from catalejo.llm import Gemini, GeminiError, from_response, to_request
from catalejo.repl import Handle, Workspace, executor, worker

OK: dict[str, Any] = {
    "candidates": [{"content": {"parts": [{"text": "listo"}], "role": "model"}}],
    "usageMetadata": {"totalTokenCount": 42},
}


def cliente(*respuestas: httpx.Response) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    """Un AsyncClient que contesta un guion y guarda lo que le pidieron."""
    vistos: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistos.append(request)
        return respuestas[min(len(vistos) - 1, len(respuestas) - 1)]

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), vistos


def cuerpo(request: httpx.Request) -> dict[str, Any]:
    datos: dict[str, Any] = json.loads(request.content)
    return datos


class TestToRequest:
    def test_el_sistema_va_en_su_campo(self) -> None:
        out = to_request((Message(Role.SYSTEM, "el preámbulo"), Message(Role.USER, "hola")))

        assert out["systemInstruction"] == {"parts": [{"text": "el preámbulo"}]}
        assert out["contents"] == [{"role": "user", "parts": [{"text": "hola"}]}]

    def test_sin_sistema_no_manda_el_campo(self) -> None:
        assert "systemInstruction" not in to_request((Message(Role.USER, "hola"),))

    def test_el_asistente_es_model(self) -> None:
        out = to_request((Message(Role.USER, "a"), Message(Role.ASSISTANT, "b")))

        assert [c["role"] for c in out["contents"]] == ["user", "model"]

    def test_dos_turnos_del_mismo_rol_se_juntan(self) -> None:
        """Pasa de verdad: `window` mete el aviso de recorte como USER justo
        después de la pregunta, que también es USER."""
        out = to_request((Message(Role.USER, "pregunta"), Message(Role.USER, "[…recorté 3]")))

        assert out["contents"] == [
            {"role": "user", "parts": [{"text": "pregunta"}, {"text": "[…recorté 3]"}]}
        ]

    def test_sin_turnos_es_error(self) -> None:
        with pytest.raises(GeminiError, match="sin turnos"):
            to_request((Message(Role.SYSTEM, "solo el preámbulo"),))


class TestFromResponse:
    def test_saca_el_texto_y_el_gasto(self) -> None:
        reply = from_response(OK)

        assert reply.message == Message(Role.ASSISTANT, "listo")
        assert reply.spent == 42

    def test_junta_las_partes(self) -> None:
        payload = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}

        assert from_response(payload).message.text == "ab"

    def test_descarta_los_resumenes_de_razonamiento(self) -> None:
        """Un razonamiento que menciona código trae backticks, y el executor
        correría eso en vez de leer el turno como respuesta final."""
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "voy a probar ```print(1)```", "thought": True},
                            {"text": "la garantía es de 24 meses"},
                        ]
                    }
                }
            ]
        }

        assert from_response(payload).message.text == "la garantía es de 24 meses"

    def test_sin_usage_el_gasto_es_cero(self) -> None:
        assert from_response({"candidates": [{"content": {"parts": [{"text": "x"}]}}]}).spent == 0

    def test_un_prompt_bloqueado_es_error(self) -> None:
        with pytest.raises(GeminiError, match="SAFETY"):
            from_response({"promptFeedback": {"blockReason": "SAFETY"}})

    def test_cortado_por_tokens_es_error(self) -> None:
        """Un bloque cercado a medio escribir no cierra, el executor lo lee como
        prosa y el agente devolvería media oración como respuesta final."""
        payload = {
            "candidates": [
                {"content": {"parts": [{"text": "```python\nprint(le"}]}, "finishReason": "MAX_TOKENS"}
            ]
        }

        with pytest.raises(GeminiError, match="MAX_TOKENS"):
            from_response(payload)

    def test_respuesta_vacia_es_error(self) -> None:
        with pytest.raises(GeminiError, match="vacío"):
            from_response({"candidates": [{"content": {"parts": [{"text": "  "}]}}]})

    def test_separa_el_corte_que_se_repite_del_que_no(self) -> None:
        """MAX_TOKENS con el mismo prompt vuelve a pasar; MALFORMED_FUNCTION_CALL
        es el parser de ellos y a la segunda suele andar."""

        def corte(razon: str) -> GeminiError:
            payload = {
                "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": razon}]
            }
            with pytest.raises(GeminiError) as e:
                from_response(payload)
            return e.value

        assert corte("MALFORMED_FUNCTION_CALL").transitorio
        assert corte("OTHER").transitorio
        assert not corte("MAX_TOKENS").transitorio
        assert not corte("SAFETY").transitorio


class TestComplete:
    async def test_pega_donde_va_y_la_key_no_va_en_la_url(self) -> None:
        client, vistos = cliente(httpx.Response(200, json=OK))
        gem = Gemini("gemini-3.8-flash", api_key="secreto", client=client)

        reply = await gem.complete((Message(Role.USER, "hola"),))

        assert reply.spent == 42
        assert str(vistos[0].url).endswith("/models/gemini-3.8-flash:generateContent")
        assert vistos[0].headers["x-goog-api-key"] == "secreto"
        assert "secreto" not in str(vistos[0].url)

    async def test_sin_opciones_no_manda_generation_config(self) -> None:
        """Los Gemini 3 vienen calibrados en temperature 1.0; mandar 0 los empeora."""
        client, vistos = cliente(httpx.Response(200, json=OK))

        await Gemini(api_key="k", client=client).complete((Message(Role.USER, "hola"),))

        assert "generationConfig" not in cuerpo(vistos[0])

    async def test_el_thinking_viaja_en_el_config(self) -> None:
        client, vistos = cliente(httpx.Response(200, json=OK))
        gem = Gemini(api_key="k", thinking="low", max_output_tokens=99, client=client)

        await gem.complete((Message(Role.USER, "hola"),))

        assert cuerpo(vistos[0])["generationConfig"] == {
            "maxOutputTokens": 99,
            "thinkingConfig": {"thinkingLevel": "low"},
        }

    async def test_reintenta_lo_transitorio(self) -> None:
        client, vistos = cliente(httpx.Response(503), httpx.Response(200, json=OK))
        gem = Gemini(api_key="k", client=client, backoff=0)

        assert (await gem.complete((Message(Role.USER, "hola"),))).spent == 42
        assert len(vistos) == 2

    async def test_no_reintenta_un_pedido_mal_armado(self) -> None:
        client, vistos = cliente(httpx.Response(400, text="modelo inexistente"))
        gem = Gemini(api_key="k", client=client, backoff=0)

        with pytest.raises(GeminiError, match="modelo inexistente"):
            await gem.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 1

    async def test_se_rinde_y_dice_por_que(self) -> None:
        client, vistos = cliente(httpx.Response(429, text="cuota"))
        gem = Gemini(api_key="k", client=client, retries=2, backoff=0)

        with pytest.raises(GeminiError, match=r"\(3 intentos\)"):
            await gem.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 3

    async def test_reintenta_un_turno_cortado_aunque_el_http_fue_200(self) -> None:
        """El caso que costó dos preguntas de veinte: la API contesta 200 y el
        turno viene roto. El reintento tiene que cubrir el parseo, no solo el HTTP."""
        roto = {
            "candidates": [{"content": {"parts": []}, "finishReason": "MALFORMED_FUNCTION_CALL"}]
        }
        client, vistos = cliente(httpx.Response(200, json=roto), httpx.Response(200, json=OK))
        gem = Gemini(api_key="k", client=client, backoff=0)

        assert (await gem.complete((Message(Role.USER, "hola"),))).spent == 42
        assert len(vistos) == 2

    async def test_un_corte_por_safety_no_se_reintenta(self) -> None:
        bloqueado = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
        client, vistos = cliente(httpx.Response(200, json=bloqueado))
        gem = Gemini(api_key="k", client=client, backoff=0)

        with pytest.raises(GeminiError, match="SAFETY"):
            await gem.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 1

    async def test_una_conexion_cortada_tambien_se_reintenta(self) -> None:
        vistos: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            vistos.append(request)
            if len(vistos) == 1:
                raise httpx.ConnectError("se cayó")
            return httpx.Response(200, json=OK)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        gem = Gemini(api_key="k", client=client, backoff=0)

        assert (await gem.complete((Message(Role.USER, "hola"),))).spent == 42

    async def test_sin_key_no_arranca(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
            Gemini()


class TestConElWorker:
    async def test_gemini_caido_es_un_fail_y_el_loop_corta(self) -> None:
        client, vistos = cliente(httpx.Response(500, text="se rompió"))
        gem = Gemini(api_key="k", client=client, retries=0, backoff=0)

        out = await worker(gem, Handle())(Log(said=(Message(Role.USER, "hola"),)))

        assert out.fails == (
            Fail("model", "GeminiError: la API no respondió (1 intentos). HTTP 500: se rompió"),
        )
        assert out.said == ()

    async def test_el_agente_entero_contra_un_transporte_falso(self) -> None:
        guion = [
            httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "```python\nprint(grep(ctx, 'garantia'))\n```"}]}},
                    ],
                    "usageMetadata": {"totalTokenCount": 100},
                },
            ),
            httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "24 meses."}]}}],
                    "usageMetadata": {"totalTokenCount": 150},
                },
            ),
        ]
        client, vistos = cliente(*guion)
        gem = Gemini(api_key="k", client=client)
        ws = Workspace("relleno\ngarantia: 24 meses desde la compra\nmás relleno")

        agente = loop(then(worker(gem, Handle()), executor(ws)), max_steps=5)
        out = await agente(Log(said=(Message(Role.USER, "cuánta garantía tiene"),)))

        assert out.said[-1].text == "24 meses."
        assert out.spent == 250
        assert out.fails == ()
        # el segundo pedido ya lleva la salida del REPL como turno de usuario
        segundo = cuerpo(vistos[1])["contents"]
        assert "garantia: 24 meses desde la compra" in json.dumps(segundo, ensure_ascii=False)


@pytest.mark.skipif(not os.environ.get("RLM_LIVE"), reason="RLM_LIVE=1 para pegarle a la API")
class TestEnVivo:
    async def test_contesta_algo(self) -> None:
        gem = Gemini(thinking="low")
        try:
            reply = await gem.complete((Message(Role.USER, "Contesta solo: ok"),))
        finally:
            await gem.aclose()

        assert reply.message.role is Role.ASSISTANT
        assert reply.spent > 0

    async def test_un_modelo_que_no_existe_es_error_claro(self) -> None:
        gem = Gemini("gemini-inventado-9")
        try:
            with pytest.raises(GeminiError, match="HTTP 404"):
                await gem.complete((Message(Role.USER, "hola"),))
        finally:
            await gem.aclose()


class TestZero:
    async def test_una_conversacion_sin_pregunta_no_llega_a_la_red(self) -> None:
        client, vistos = cliente(httpx.Response(200, json=OK))

        out = await worker(Gemini(api_key="k", client=client), Handle())(ZERO)

        assert vistos == []
        assert out.fails[0].who == "model"
