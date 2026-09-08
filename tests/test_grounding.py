from catalejo.core import ZERO, Cell, Log, Message, Role, Status, loop, merge, then
from catalejo.llm import Stub
from catalejo.repl import Handle, Workspace, executor, grounded, worker
from catalejo.repl.grounding import AVISO, SIN_FUNDAMENTO

PAYLOAD = "manual\ngarantia: 24 meses\nfin"


def dicho(texto: str, role: Role = Role.ASSISTANT) -> Message:
    return Message(role, texto)


def bloque(code: str) -> str:
    return f"```python\n{code}\n```"


AVISADO = Message(Role.USER, AVISO.format(var="ctx"))


class TestGrounded:
    async def test_no_opina_mientras_escribe_codigo(self) -> None:
        seen = Log(said=(dicho(bloque("print(1)")),))

        assert await grounded()(seen) == ZERO

    async def test_no_opina_si_ya_consulto(self) -> None:
        seen = Log(said=(dicho("la garantía es de 24 meses"),), reads=1)

        assert await grounded()(seen) == ZERO

    async def test_no_opina_sobre_un_log_vacio(self) -> None:
        assert await grounded()(ZERO) == ZERO

    async def test_no_opina_si_el_ultimo_no_es_del_modelo(self) -> None:
        seen = Log(said=(dicho("[repl] salida:\n3", Role.USER),))

        assert await grounded()(seen) == ZERO

    async def test_veta_la_respuesta_que_no_consulto_nada(self) -> None:
        seen = Log(said=(dicho("la garantía es de 24 meses"),))

        out = await grounded()(seen)

        assert out.vote is Status.CONTINUE
        assert out.said == (AVISADO,)
        assert "no sale de `ctx` sino de tu memoria" in out.said[0].text

    async def test_el_aviso_nombra_la_variable_real(self) -> None:
        out = await grounded("tickets")(Log(said=(dicho("son 1768"),)))

        assert "`tickets`" in out.said[0].text

    async def test_el_veto_le_gana_al_done_del_executor(self) -> None:
        """No puede borrar el DONE de otra célula, y no le hace falta: `vote` se
        junta por máximo y CONTINUE es mayor. Para esto estaba el orden."""
        del_executor = Log(vote=Status.DONE)
        del_grounding = await grounded()(Log(said=(dicho("de memoria"),)))

        assert merge(del_executor, del_grounding).vote is Status.CONTINUE

    async def test_avisa_una_sola_vez_y_despues_lo_deja_terminar(self) -> None:
        """Insistir hasta el tope serían doce llamadas para la misma conclusión."""
        seen = Log(said=(dicho("de memoria"), AVISADO, dicho("igual de memoria")))

        out = await grounded()(seen)

        assert out.fails == (SIN_FUNDAMENTO,)
        assert out.vote is Status.QUIET  # deja que el DONE del executor mande
        assert out.said == ()


class TestElExecutorCuentaLasConsultas:
    async def test_una_corrida_que_devolvio_algo_cuenta(self) -> None:
        ws = Workspace(PAYLOAD)

        out = await executor(ws)(Log(said=(dicho(bloque("print(len(ctx))")),)))

        assert out.reads == 1

    async def test_un_snippet_mudo_no_cuenta(self) -> None:
        """No imprimió nada, así que no le enseñó nada."""
        ws = Workspace(PAYLOAD)

        out = await executor(ws)(Log(said=(dicho(bloque("x = 1")),)))

        assert out.reads == 0

    async def test_la_respuesta_final_no_cuenta(self) -> None:
        ws = Workspace(PAYLOAD)

        out = await executor(ws)(Log(said=(dicho("ya está, son 24 meses"),)))

        assert out.reads == 0
        assert out.vote is Status.DONE


class TestLaFabricacion:
    """La corrida real que destapó esto: el modelo pidió `import re`, se comió el
    error y contestó el índice de 80 archivos de memoria, inventando nombres que
    no existen. Votó DONE y no reportó un solo fail."""

    def agente(self, model: Stub, ws: Workspace) -> Cell:
        return loop(
            then(worker(model, Handle(tools=ws.tools)), executor(ws), grounded()),
            max_steps=6,
        )

    async def test_sin_grounding_la_mentira_pasa_como_exito(self) -> None:
        model = Stub(bloque("import re"), "el índice es: action.go, actor.go, alias.go")
        ws = Workspace(PAYLOAD)

        agente = loop(then(worker(model, Handle()), executor(ws)), max_steps=6)
        out = await agente(Log(said=(dicho("dame el índice", Role.USER),)))

        assert out.vote is Status.DONE
        assert out.fails == ()  # el sistema dice que salió bien

    async def test_con_grounding_lo_manda_a_consultar(self) -> None:
        model = Stub(
            bloque("import re"),
            "el índice es: action.go, actor.go, alias.go",
            bloque("print(grep(ctx, 'garantia'))"),
            "la garantía es de 24 meses",
        )
        ws = Workspace(PAYLOAD)
        out = await self.agente(model, ws)(Log(said=(dicho("dame el índice", Role.USER),)))

        assert out.said[-1].text == "la garantía es de 24 meses"
        assert out.vote is Status.DONE
        assert out.fails == ()
        assert out.reads == 1
        assert AVISADO in out.said

    async def test_si_insiste_la_respuesta_sale_etiquetada(self) -> None:
        model = Stub(
            bloque("import re"),
            "action.go, actor.go, alias.go",
            "en serio: action.go, actor.go, alias.go",
        )
        ws = Workspace(PAYLOAD)
        out = await self.agente(model, ws)(Log(said=(dicho("dame el índice", Role.USER),)))

        assert out.fails == (SIN_FUNDAMENTO,)
        assert out.vote is Status.DONE  # sale, pero sale con la etiqueta puesta
        assert out.reads == 0

    async def test_no_molesta_a_una_corrida_normal(self) -> None:
        model = Stub(bloque("print(grep(ctx, 'garantia'))"), "la garantía es de 24 meses")
        ws = Workspace(PAYLOAD)
        out = await self.agente(model, ws)(Log(said=(dicho("cuánta garantía", Role.USER),)))

        assert out.said[-1].text == "la garantía es de 24 meses"
        assert out.fails == ()
        assert AVISADO not in out.said
