from catalejo.core import ZERO, Cell, Fail, Log, Message, Role, Status, loop, then
from catalejo.repl import (
    Output,
    Stub,
    Workspace,
    executor,
    extract_code,
    otro_bloque,
    render,
)


def dicho(texto: str, role: Role = Role.ASSISTANT) -> Log:
    return Log(said=(Message(role, texto),))


class TestExtractCode:
    def test_bloque_con_etiqueta_de_lenguaje(self) -> None:
        texto = "Voy a mirar.\n```python\nprint(len(ctx))\n```"

        assert extract_code(texto) == "print(len(ctx))"

    def test_bloque_sin_etiqueta(self) -> None:
        assert extract_code("```\nprint(1)\n```") == "print(1)"

    def test_bloque_de_una_linea(self) -> None:
        assert extract_code("```print(1)```") == "print(1)"

    def test_bloque_de_una_linea_seguido_de_prosa(self) -> None:
        """El salto de línea de la prosa no es la etiqueta de lenguaje."""
        assert extract_code("```print(1)```\ny después sigo.") == "print(1)"

    def test_sin_cerca_es_una_respuesta_final(self) -> None:
        assert extract_code("El precio es Q475.00.") == ""

    def test_una_cerca_sin_cerrar_nunca_corre(self) -> None:
        """Un bloque a medias, por backticks sueltos o una respuesta cortada, no
        puede correr como código."""
        assert extract_code("Mirá esto:\n```python\nprint(len(ctx))") == ""

    def test_vacio(self) -> None:
        assert extract_code("") == ""


class TestOtroBloque:
    def test_uno_solo_no_sobra_nada(self) -> None:
        assert not otro_bloque("```\nprint(1)\n```")

    def test_dos_bloques(self) -> None:
        assert otro_bloque("```\nprint(1)\n```\ny después\n```\nprint(2)\n```")

    def test_la_prosa_de_después_no_es_un_bloque(self) -> None:
        assert not otro_bloque("```print(1)```\ny después sigo.")

    def test_una_cerca_sin_cerrar_al_final_no_cuenta(self) -> None:
        """No va a correr igual, así que avisar de eso sería ruido."""
        assert not otro_bloque("```\nprint(1)\n```\ny\n```python\nprint(2)")


class TestRender:
    def test_la_salida(self) -> None:
        assert render(Output(stdout="42\n")) == "[repl] salida:\n42"

    def test_el_error(self) -> None:
        out = Output(stdout="voy bien\n", err="NameError: x")

        assert render(out) == "[repl] salida:\nvoy bien\nerror: NameError: x"

    def test_cuando_no_imprimio_nada_lo_dice(self) -> None:
        """Si no, el modelo no distingue "corrió y no imprimió" de "no corrió"."""
        assert render(Output()) == "[repl] salida:\n(el snippet no imprimió nada)"


class TestExecutor:
    async def test_corre_el_codigo_de_la_ultima_propuesta(self) -> None:
        env = Stub()

        out = await executor(env)(dicho("```\nprint(1)\n```"))

        assert env.corrido == ["print(1)"]
        assert out.said[0].text == "[repl] salida:\neco: print(1)"

    async def test_la_salida_vuelve_como_turno_del_usuario(self) -> None:
        out = await executor(Stub())(dicho("```\nprint(1)\n```"))

        assert out.said[0].role is Role.USER

    async def test_con_codigo_vota_seguir(self) -> None:
        out = await executor(Stub())(dicho("```\nprint(1)\n```"))

        assert out.vote is Status.CONTINUE

    async def test_sin_codigo_vota_terminar_y_no_corre_nada(self) -> None:
        env = Stub()

        out = await executor(env)(dicho("El precio es Q475.00."))

        assert out.vote is Status.DONE
        assert out.said == ()
        assert env.corrido == []

    async def test_sin_propuesta_del_modelo_no_opina(self) -> None:
        """QUIET y no DONE: DONE diría que el modelo contestó, y acá no habló
        nadie. El loop corta con los dos, pero solo uno es verdad."""
        assert await executor(Stub())(ZERO) == ZERO

    async def test_no_corre_lo_que_no_dijo_el_modelo(self) -> None:
        """La salida del REPL trae texto del contexto, que es no confiable, y ese
        texto puede tener un bloque cercado adentro. Cuando el worker se cae no
        agrega nada y esa salida queda como último dicho: mirarla sin mirar el rol
        es ejecutar el corpus."""
        env = Stub()
        del_corpus = "[repl] salida:\n```python\nprint('del corpus')\n```"

        out = await executor(env)(dicho(del_corpus, Role.USER))

        assert env.corrido == []
        assert out == ZERO

    async def test_avisa_cuando_habia_mas_de_un_bloque(self) -> None:
        """Corre uno por turno, y callarse el resto es el mismo defecto que el
        grep que recortaba en silencio: el modelo pidió dos cosas, vio una salida
        y no tiene cómo saber que la otra nunca pasó."""
        env = Stub()

        out = await executor(env)(dicho("```\nprint(1)\n```\ny después\n```\nprint(2)\n```"))

        assert env.corrido == ["print(1)"]
        assert "solo corrió el primero" in out.said[0].text

    async def test_un_snippet_que_revienta_no_es_un_fail(self) -> None:
        """Es flujo normal del REPL: el modelo ve el error y lo corrige en el
        bloque siguiente."""
        env = Stub(lambda code: Output(err="NameError: no_existe"))

        out = await executor(env)(dicho("```\nprint(no_existe)\n```"))

        assert out.fails == ()
        assert out.vote is Status.CONTINUE
        assert "error: NameError: no_existe" in out.said[0].text

    async def test_un_environment_caido_si_es_un_fail(self) -> None:
        """Y el voto queda en QUIET, para que el loop corte en vez de seguir
        pidiéndole código a un REPL muerto."""

        class Muerto:
            async def run(self, code: str) -> Output:
                raise ConnectionError("sandbox no responde")

        out = await executor(Muerto())(dicho("```\nprint(1)\n```"))

        assert out.fails == (Fail("repl", "ConnectionError: sandbox no responde"),)
        assert out.vote is Status.QUIET
        assert out.said == ()


def worker_guion(*mensajes: str) -> Cell:
    """Un worker de mentira que dice lo que le toca, un mensaje por paso."""
    paso = 0

    async def cell(seen: Log) -> Log:
        nonlocal paso
        texto = mensajes[paso]
        paso += 1
        return Log(said=(Message(Role.ASSISTANT, texto),))

    return cell


class TestElLoopEntero:
    async def test_el_modelo_consulta_el_contexto_sin_leerlo(self) -> None:
        payload = "linea uno\nlinea dos\nprecio: Q475.00"
        ws = Workspace(payload)
        worker = worker_guion(
            "Veo el tamaño.\n```python\nprint(len(ctx))\n```",
            "Busco el precio.\n```python\nprint(grep(ctx, 'precio'))\n```",
            "El precio es Q475.00.",
        )

        agente = loop(then(worker, executor(ws)), max_steps=10)
        out = await agente(dicho("cuánto sale", Role.USER))

        dichos = [m.text for m in out.said]
        assert len(dichos) == 5
        assert str(len(payload)) in dichos[1]
        assert "3: precio: Q475.00" in dichos[3]
        assert dichos[4] == "El precio es Q475.00."
        assert out.vote is Status.DONE
        assert out.fails == ()

    async def test_un_worker_caido_no_corre_la_salida_anterior(self) -> None:
        """El caso de arriba, pero armado como pasa de verdad: el modelo revienta,
        el worker no agrega nada, y el último dicho es la salida vieja del REPL."""
        env = Stub()

        async def worker_caido(seen: Log) -> Log:
            return Log(fails=(Fail("model", "503"),))

        agente = loop(then(worker_caido, executor(env)), max_steps=5)
        out = await agente(dicho("[repl] salida:\n```python\nprint('del corpus')\n```", Role.USER))

        assert env.corrido == []
        assert out.vote is Status.QUIET  # el loop corta en vez de repetirlo cinco veces
        assert out.fails == (Fail("model", "503"),)

    async def test_el_modelo_se_recupera_de_su_propio_error(self) -> None:
        ws = Workspace("hola")
        worker = worker_guion(
            "```python\nprint(no_existe)\n```",
            "Perdón, corrijo.\n```python\nprint(ctx)\n```",
            "Dice hola.",
        )

        out = await loop(then(worker, executor(ws)), max_steps=10)(ZERO)

        assert "error: NameError" in out.said[1].text
        assert out.said[3].text == "[repl] salida:\nhola"
        assert out.fails == ()
