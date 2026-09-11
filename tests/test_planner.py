"""La célula del plan: qué admite, qué cierra, y cuándo deja terminar el turno."""

from catalejo.core import ZERO, Log, Message, PlanOp, Role, Status, merge, proyectar, then
from catalejo.llm import Stub
from catalejo.repl import Handle, Registro, Verbos, Workspace, executor, planner, worker
from catalejo.repl.planner import ESTADO, FALTAN, INCOMPLETO

PAYLOAD = "manual\ngarantia: 24 meses\nfin"

LEYO: Registro = {"leyo": lambda log: log.reads > 0}

SEMILLA = (PlanOp("add_step", "buscar", "encontrar el dato"),)


def dicho(texto: str, role: Role = Role.ASSISTANT) -> Message:
    return Message(role, texto)


def bloque(code: str) -> str:
    return f"```python\n{code}\n```"


def avisado(log: Log) -> Log:
    """Un Log que ya se llevó el empujón, para probar la segunda vuelta."""
    return merge(log, Log(said=(dicho(f"{FALTAN} `buscar`.", Role.USER),)))


class TestLaSemilla:
    async def test_entra_en_el_primer_paso(self) -> None:
        out = await planner(Verbos(), {}, semilla=SEMILLA)(Log(said=(dicho(bloque("x=1")),)))

        assert out.steps == SEMILLA
        assert ESTADO in out.said[0].text

    async def test_no_vuelve_a_entrar_cuando_el_canal_ya_la_tiene(self) -> None:
        seen = Log(said=(dicho(bloque("x=1")),), steps=SEMILLA)

        out = await planner(Verbos(), {}, semilla=SEMILLA)(seen)

        assert out == ZERO


class TestLaTerminacion:
    async def test_prosa_con_pasos_abiertos_pide_seguir(self) -> None:
        seen = Log(said=(dicho("la garantía es de 24 meses"),), steps=SEMILLA)

        out = await planner(Verbos(), {})(seen)

        assert out.vote is Status.CONTINUE
        assert FALTAN in out.said[0].text
        assert "`buscar`" in out.said[0].text

    async def test_el_veto_le_gana_al_done_del_executor(self) -> None:
        """Mismo mecanismo que `grounded`: no borra el DONE de nadie, opina más
        fuerte, porque `vote` se junta por máximo y CONTINUE es mayor."""
        seen = Log(said=(dicho("de memoria"),), steps=SEMILLA)

        out = await planner(Verbos(), {})(seen)

        assert merge(Log(vote=Status.DONE), out).vote is Status.CONTINUE

    async def test_con_el_plan_completo_no_toca_el_voto(self) -> None:
        """Así el turno termina cuando lo dice el predicado, y esto no agrega
        ninguna forma nueva de terminar: levanta el veto y nada más."""
        seen = Log(
            said=(dicho("la garantía es de 24 meses"),),
            steps=(*SEMILLA, PlanOp("mark", "buscar", status="done")),
        )

        out = await planner(Verbos(), {})(seen)

        assert out.vote is Status.QUIET
        assert out.said == ()
        assert merge(Log(vote=Status.DONE), out).vote is Status.DONE

    async def test_avisa_una_sola_vez_y_despues_lo_deja_terminar(self) -> None:
        seen = avisado(Log(said=(dicho("de memoria"),), steps=SEMILLA))
        seen = merge(seen, Log(said=(dicho("igual de memoria"),)))

        out = await planner(Verbos(), {})(seen)

        assert out.fails == (INCOMPLETO,)
        assert out.vote is Status.QUIET
        assert out.said == ()

    async def test_mientras_escribe_codigo_no_opina_de_la_terminacion(self) -> None:
        seen = Log(said=(dicho(bloque("print(1)")),), steps=SEMILLA)

        out = await planner(Verbos(), {})(seen)

        assert out.vote is Status.QUIET


class TestLasCompuertas:
    async def test_la_compuerta_cierra_el_paso_sin_que_el_modelo_lo_pida(self) -> None:
        semilla = (PlanOp("add_step", "buscar", completes_when="leyo"),)
        seen = Log(said=(dicho(bloque("print(1)")),), reads=1)

        out = await planner(Verbos(), LEYO, semilla=semilla)(seen)

        assert proyectar(out.steps)[0].status == "done"

    async def test_sin_el_hecho_el_paso_queda_abierto(self) -> None:
        semilla = (PlanOp("add_step", "buscar", completes_when="leyo"),)
        seen = Log(said=(dicho(bloque("print(1)")),))

        out = await planner(Verbos(), LEYO, semilla=semilla)(seen)

        assert proyectar(out.steps)[0].status == "todo"

    async def test_una_compuerta_que_no_esta_en_el_registro_no_cierra_y_se_anota(self) -> None:
        """Cierra por el lado seguro: el paso no se cierra nunca, el plan no se
        completa y el turno sale con `Fail` en vez de salir como si nada."""
        semilla = (PlanOp("add_step", "buscar", completes_when="leyó"),)

        out = await planner(Verbos(), LEYO, semilla=semilla)(Log(said=(dicho(bloque("x=1")),)))

        assert out.fails and "no está en el registro" in out.fails[0].reason
        assert proyectar(out.steps)[0].status == "todo"

    async def test_un_predicado_que_revienta_cuenta_como_no_cumplido(self) -> None:
        def explota(log: Log) -> bool:
            raise RuntimeError("bug del host")

        semilla = (PlanOp("add_step", "buscar", completes_when="leyo"),)

        out = await planner(Verbos(), {"leyo": explota}, semilla=semilla)(
            Log(said=(dicho(bloque("x=1")),))
        )

        assert out.fails and "RuntimeError" in out.fails[0].reason
        assert proyectar(out.steps)[0].status == "todo"


class TestLoQueElModeloPropone:
    async def test_los_ops_del_snippet_llegan_al_canal(self) -> None:
        v = Verbos()
        ws = Workspace(PAYLOAD, extra=v.builtins)
        code = bloque('add_step("dosis", "traer la dosis")\nprint(len(ctx))')
        model = Stub(code)
        paso = then(worker(model, Handle(tools=ws.tools)), executor(ws), planner(v, {}))

        out = await paso(Log(said=(dicho("cuánta garantía", Role.USER),)))

        assert proyectar(out.steps) == proyectar((PlanOp("add_step", "dosis", "traer la dosis"),))
        assert "dosis" in out.said[-1].text

    async def test_el_rechazo_vuelve_con_su_motivo(self) -> None:
        v = Verbos()
        v.mark("buscar", "done")

        out = await planner(v, {})(Log(said=(dicho(bloque("x=1")),)))

        assert "rechazado" in out.said[0].text
        assert "no cambia nada del plan" in out.said[0].text

    async def test_un_done_del_modelo_sobre_un_paso_con_compuerta_no_entra(self) -> None:
        v = Verbos()
        v.mark("buscar", "done")
        semilla = (PlanOp("add_step", "buscar", completes_when="leyo"),)

        out = await planner(v, LEYO, semilla=semilla)(Log(said=(dicho(bloque("x=1")),)))

        assert proyectar(out.steps)[0].status == "todo"
        assert "lo cierro yo" in out.said[0].text
