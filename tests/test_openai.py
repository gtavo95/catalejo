import json
import os
from typing import Any

import httpx
import pytest

from catalejo.core import ZERO, Fail, Log, Message, Role, loop, then
from catalejo.llm import OpenAI, OpenAIError, from_completion, to_messages
from catalejo.repl import Handle, Workspace, executor, worker

OK: dict[str, Any] = {
    "choices": [{"message": {"role": "assistant", "content": "listo"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 42},
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


def turno(**campos: Any) -> dict[str, Any]:
    mensaje = {"role": "assistant", "content": campos.pop("content", None)}
    mensaje.update({k: v for k, v in campos.items() if k == "refusal"})
    return {
        "choices": [
            {"message": mensaje, "finish_reason": campos.get("finish_reason", "stop")}
        ]
    }


class TestToMessages:
    def test_el_sistema_es_un_mensaje_mas(self) -> None:
        """La diferencia con Gemini, que lo quiere en un campo aparte."""
        out = to_messages((Message(Role.SYSTEM, "el preámbulo"), Message(Role.USER, "hola")))

        assert out == [
            {"role": "system", "content": "el preámbulo"},
            {"role": "user", "content": "hola"},
        ]

    def test_dos_turnos_del_mismo_rol_van_como_vinieron(self) -> None:
        """`window` mete el aviso de recorte como USER justo después de la
        pregunta, que también es USER. Gemini obliga a juntarlos, acá no."""
        out = to_messages((Message(Role.USER, "pregunta"), Message(Role.USER, "[…recorté 3]")))

        assert [m["content"] for m in out] == ["pregunta", "[…recorté 3]"]

    def test_sin_turnos_es_error(self) -> None:
        with pytest.raises(OpenAIError, match="sin turnos"):
            to_messages((Message(Role.SYSTEM, "solo el preámbulo"),))


class TestFromCompletion:
    def test_saca_el_texto_y_el_gasto(self) -> None:
        reply = from_completion(OK)

        assert reply.message == Message(Role.ASSISTANT, "listo")
        assert reply.spent == 42

    def test_sin_usage_el_gasto_es_cero(self) -> None:
        assert from_completion(turno(content="x")).spent == 0

    def test_sin_choices_es_error(self) -> None:
        with pytest.raises(OpenAIError, match="sin choices"):
            from_completion({"usage": {"total_tokens": 1}})

    def test_un_error_en_el_cuerpo_se_lee(self) -> None:
        """Algunos clones del contrato devuelven 200 con el error adentro."""
        payload = {"error": {"message": "modelo inexistente", "type": "invalid_request_error"}}

        with pytest.raises(OpenAIError, match="modelo inexistente"):
            from_completion(payload)

    def test_una_negativa_se_dice_con_su_texto(self) -> None:
        """Viene en su propio campo con el content vacío. Sin leerlo, esto sería
        indistinguible de un turno vacío, que sí se reintenta."""
        with pytest.raises(OpenAIError, match="se negó: no puedo ayudar"):
            from_completion(turno(content="", refusal="no puedo ayudar con eso"))

    def test_cortado_por_length_es_error(self) -> None:
        """Un bloque cercado a medio escribir no cierra, el executor lo lee como
        prosa y el agente devolvería media oración como respuesta final."""
        payload = turno(content="```python\nprint(le", finish_reason="length")

        with pytest.raises(OpenAIError, match="length"):
            from_completion(payload)

    def test_respuesta_vacia_es_error(self) -> None:
        with pytest.raises(OpenAIError, match="vacío"):
            from_completion(turno(content="  "))

    def test_un_content_nulo_tambien(self) -> None:
        """El agujero de los modelos de razonamiento: 200 bien formado, `content`
        en null, todo el presupuesto de salida gastado pensando."""
        with pytest.raises(OpenAIError, match="razonamiento"):
            from_completion(turno(content=None))

    def test_separa_el_corte_que_se_repite_del_que_no(self) -> None:
        """Este repo no declara herramientas, así que un corte por `tool_calls`
        es el parser de ellos y a la segunda suele andar. `length` vuelve a
        cortar en el mismo lugar con el mismo prompt."""

        def corte(razon: str) -> OpenAIError:
            with pytest.raises(OpenAIError) as e:
                from_completion(turno(content="x", finish_reason=razon))
            return e.value

        assert corte("tool_calls").transitorio
        assert not corte("length").transitorio
        assert not corte("content_filter").transitorio

    def test_un_turno_vacio_se_reintenta(self) -> None:
        with pytest.raises(OpenAIError) as e:
            from_completion(turno(content=""))

        assert e.value.transitorio


class TestComplete:
    async def test_pega_donde_va_y_la_key_va_en_el_header(self) -> None:
        client, vistos = cliente(httpx.Response(200, json=OK))
        oai = OpenAI("gpt-5.1", api_key="secreto", client=client)

        reply = await oai.complete((Message(Role.USER, "hola"),))

        assert reply.spent == 42
        assert str(vistos[0].url) == "https://api.openai.com/v1/chat/completions"
        assert vistos[0].headers["authorization"] == "Bearer secreto"
        assert "secreto" not in str(vistos[0].url)
        assert cuerpo(vistos[0])["model"] == "gpt-5.1"

    async def test_sin_opciones_manda_solo_modelo_y_mensajes(self) -> None:
        """Los modelos de razonamiento rechazan cualquier temperature que no sea 1."""
        client, vistos = cliente(httpx.Response(200, json=OK))

        await OpenAI(api_key="k", client=client).complete((Message(Role.USER, "hola"),))

        assert set(cuerpo(vistos[0])) == {"model", "messages"}

    async def test_el_thinking_viaja_como_reasoning_effort(self) -> None:
        """El mismo parámetro que en Gemini, para que cambiar de proveedor sea
        cambiar la clase y nada más."""
        client, vistos = cliente(httpx.Response(200, json=OK))
        oai = OpenAI(api_key="k", thinking="low", max_output_tokens=99, client=client)

        await oai.complete((Message(Role.USER, "hola"),))

        enviado = cuerpo(vistos[0])
        assert enviado["reasoning_effort"] == "low"
        assert enviado["max_completion_tokens"] == 99
        assert "max_tokens" not in enviado

    async def test_otro_proveedor_es_cambiar_la_base(self) -> None:
        """Cualquier endpoint que hable Chat Completions entra por acá."""
        client, vistos = cliente(httpx.Response(200, json=OK))
        oai = OpenAI("llama-3.3-70b", base_url="http://localhost:11434/v1/", client=client)

        await oai.complete((Message(Role.USER, "hola"),))

        assert str(vistos[0].url) == "http://localhost:11434/v1/chat/completions"
        assert "authorization" not in vistos[0].headers

    async def test_reintenta_lo_transitorio(self) -> None:
        client, vistos = cliente(httpx.Response(503), httpx.Response(200, json=OK))
        oai = OpenAI(api_key="k", client=client, backoff=0)

        assert (await oai.complete((Message(Role.USER, "hola"),))).spent == 42
        assert len(vistos) == 2

    async def test_no_reintenta_un_pedido_mal_armado(self) -> None:
        client, vistos = cliente(httpx.Response(400, text="modelo inexistente"))
        oai = OpenAI(api_key="k", client=client, backoff=0)

        with pytest.raises(OpenAIError, match="modelo inexistente"):
            await oai.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 1

    async def test_se_rinde_y_dice_por_que(self) -> None:
        client, vistos = cliente(httpx.Response(429, text="cuota"))
        oai = OpenAI(api_key="k", client=client, retries=2, backoff=0)

        with pytest.raises(OpenAIError, match=r"\(3 intentos\)\. HTTP 429"):
            await oai.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 3

    async def test_reintenta_un_turno_vacio_aunque_el_http_fue_200(self) -> None:
        """El caso que motivó todo esto: la API contesta 200 y el turno viene sin
        texto. El reintento tiene que cubrir el parseo, no solo el HTTP."""
        client, vistos = cliente(
            httpx.Response(200, json=turno(content=None)), httpx.Response(200, json=OK)
        )
        oai = OpenAI(api_key="k", client=client, backoff=0)

        assert (await oai.complete((Message(Role.USER, "hola"),))).spent == 42
        assert len(vistos) == 2

    async def test_una_negativa_no_se_reintenta(self) -> None:
        negado = turno(content="", refusal="no puedo ayudar con eso")
        client, vistos = cliente(httpx.Response(200, json=negado))
        oai = OpenAI(api_key="k", client=client, backoff=0)

        with pytest.raises(OpenAIError, match="se negó"):
            await oai.complete((Message(Role.USER, "hola"),))
        assert len(vistos) == 1

    async def test_una_conexion_cortada_tambien_se_reintenta(self) -> None:
        vistos: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            vistos.append(request)
            if len(vistos) == 1:
                raise httpx.ConnectError("se cayó")
            return httpx.Response(200, json=OK)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        oai = OpenAI(api_key="k", client=client, backoff=0)

        assert (await oai.complete((Message(Role.USER, "hola"),))).spent == 42

    async def test_sin_key_no_arranca(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(OpenAIError, match="OPENAI_API_KEY"):
            OpenAI()

    async def test_un_endpoint_local_no_pide_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Obligar a inventarse una key contra vLLM sería un trámite por nada."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert OpenAI(base_url="http://localhost:8000/v1").model


class TestConElWorker:
    async def test_la_api_caida_es_un_fail_y_el_loop_corta(self) -> None:
        client, vistos = cliente(httpx.Response(500, text="se rompió"))
        oai = OpenAI(api_key="k", client=client, retries=0, backoff=0)

        out = await worker(oai, Handle())(Log(said=(Message(Role.USER, "hola"),)))

        assert out.fails == (
            Fail("model", "OpenAIError: la API no respondió (1 intentos). HTTP 500: se rompió"),
        )
        assert out.said == ()

    async def test_el_agente_entero_contra_un_transporte_falso(self) -> None:
        """El contexto grande NUNCA entra al prompt: entra por el REPL."""
        guion = [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "```python\nprint(grep(ctx, 'garantia'))\n```"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 100},
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "24 meses."}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 150},
                },
            ),
        ]
        client, vistos = cliente(*guion)
        oai = OpenAI(api_key="k", client=client)
        ws = Workspace("relleno\ngarantia: 24 meses desde la compra\nmás relleno")

        agente = loop(then(worker(oai, Handle()), executor(ws)), max_steps=5)
        out = await agente(Log(said=(Message(Role.USER, "cuánta garantía tiene"),)))

        assert out.said[-1].text == "24 meses."
        assert out.spent == 250
        assert out.fails == ()
        primero = json.dumps(cuerpo(vistos[0]), ensure_ascii=False)
        assert "garantia: 24 meses desde la compra" not in primero
        segundo = json.dumps(cuerpo(vistos[1]), ensure_ascii=False)
        assert "garantia: 24 meses desde la compra" in segundo


class TestZero:
    async def test_una_conversacion_sin_pregunta_no_llega_a_la_red(self) -> None:
        client, vistos = cliente(httpx.Response(200, json=OK))

        out = await worker(OpenAI(api_key="k", client=client), Handle())(ZERO)

        assert vistos == []
        assert out.fails[0].who == "model"


@pytest.mark.skipif(not os.environ.get("RLM_LIVE"), reason="RLM_LIVE=1 para pegarle a la API")
class TestEnVivo:
    async def test_contesta_algo(self) -> None:
        oai = OpenAI(thinking="low")
        try:
            reply = await oai.complete((Message(Role.USER, "Contesta solo: ok"),))
        finally:
            await oai.aclose()

        assert reply.message.role is Role.ASSISTANT
        assert reply.spent > 0

    async def test_un_modelo_que_no_existe_es_error_claro(self) -> None:
        oai = OpenAI("gpt-inventado-9")
        try:
            with pytest.raises(OpenAIError, match="HTTP 40"):
                await oai.complete((Message(Role.USER, "hola"),))
        finally:
            await oai.aclose()
