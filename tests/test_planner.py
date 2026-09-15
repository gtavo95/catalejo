"""La célula del plan: qué admite, qué cierra, y cuándo deja terminar el turno."""

from catalejo.core import ZERO, Log, Message, PlanOp, Role, Status, merge, proyectar, then
from catalejo.llm import Stub
from catalejo.rlm import (
    Handle,
    Registro,
    Verbos,
    Workspace,
    avanzar,
    derivar,
    executor,
    exigir,
    planner,
    worker,
)
from catalejo.rlm.celulas.planner import ESTADO, FALTAN, INCOMPLETO, render_plan

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


class TestAvanzar:
    async def test_no_vota_ni_con_prosa_y_pasos_abiertos(self) -> None:
        out = await avanzar(Verbos(), {}, semilla=SEMILLA)(Log(said=(dicho("de memoria"),)))

        assert out.vote is Status.QUIET
        assert out.steps == SEMILLA
        assert out.said == ()

    async def test_con_el_canal_sembrado_por_el_host_no_reemite_la_semilla(self) -> None:
        """Así entra el plan de la conversación con su historia en cada turno."""
        seen = Log(said=(dicho(bloque("x=1")),), steps=SEMILLA)

        assert await avanzar(Verbos(), {}, semilla=SEMILLA)(seen) == ZERO

    async def test_sobre_course_proyecta_y_emite_en_course(self) -> None:
        semilla = (PlanOp("add_step", "plaga", completes_when="leyo"),)
        seen = Log(said=(dicho(bloque("x=1")),), reads=1)

        out = await avanzar(Verbos(), LEYO, canal="course", semilla=semilla)(seen)

        assert out.steps == ()
        assert proyectar(out.course)[0].status == "done"

    async def test_con_etiqueta_toma_lo_suyo_y_deja_el_resto(self) -> None:
        v = Verbos()
        v.mark("plaga", "active")
        v.mark("buscar", "active")
        v.add_step("garantia", "preguntar", en="venta")
        v.add_step("sintomas", "qué ve", padre="plaga", en="diagnostico")
        v.add_step("plaga", "identificar la plaga")
        v.add_step("sintoma", "qué ve", padre="plaga")
        v.mark("sintoma", "active")
        seen = Log(
            said=(dicho(bloque("x=1")),),
            course=(PlanOp("add_step", "plaga"),),
            steps=(PlanOp("add_step", "buscar"),),
        )

        venta = await avanzar(v, {}, canal="course", etiqueta="venta")(seen)
        turno = await avanzar(v, {})(merge(seen, venta))

        assert [op.id for op in venta.course] == ["plaga", "garantia", "sintomas", "sintoma", "sintoma"]
        assert [op.id for op in turno.steps] == ["buscar"]
        assert v.tomar() == ()
        assert "rechazado add_step(\"plaga\"" in venta.said[0].text

    async def test_sin_dibujar_solo_habla_para_rechazar(self) -> None:
        v = Verbos()
        seen = Log(said=(dicho(bloque("x=1")),), course=(PlanOp("add_step", "plaga"),))

        callado = await avanzar(v, {}, canal="course", etiqueta="venta", dibuja=False)(seen)
        v.skip("plaga")
        rechaza = await avanzar(v, {}, canal="course", flex="firme", etiqueta="venta", dibuja=False)(seen)

        assert callado == ZERO
        assert rechaza.said[0].text.startswith("rechazado skip") and rechaza.course == ()


class TestExigir:
    async def test_veta_la_prosa_con_pasos_abiertos(self) -> None:
        out = await exigir()(Log(said=(dicho("de memoria"),), steps=SEMILLA))

        assert out.vote is Status.CONTINUE
        assert FALTAN in out.said[0].text

    async def test_con_el_plan_completo_no_dice_nada(self) -> None:
        seen = Log(said=(dicho("listo"),), steps=(*SEMILLA, PlanOp("mark", "buscar", status="done")))

        assert await exigir()(seen) == ZERO

    async def test_una_checklist_vacia_no_exige_nada(self) -> None:
        """La diferencia con `completo`: un turno sin checklist termina cuando el
        modelo lo dice, y ese turno existe (la hoja `plaga` no exige nada)."""
        assert await exigir()(Log(said=(dicho("¿qué plaga ves?"),))) == ZERO

    async def test_ya_avisado_anota_incompleto(self) -> None:
        seen = avisado(Log(said=(dicho("de memoria"),), steps=SEMILLA))
        seen = merge(seen, Log(said=(dicho("igual"),)))

        out = await exigir()(seen)

        assert out.fails == (INCOMPLETO,) and out.vote is Status.QUIET

    async def test_sobre_codigo_no_opina(self) -> None:
        assert await exigir()(Log(said=(dicho(bloque("x=1")),), steps=SEMILLA)) == ZERO


class TestPlannerEsLosDos:
    async def test_planner_es_then_de_avanzar_y_exigir(self) -> None:
        semilla = (PlanOp("add_step", "buscar", completes_when="leyo"),)
        for log in (
            Log(said=(dicho("de memoria"),), steps=semilla),
            Log(said=(dicho(bloque("x=1")),), reads=1),
        ):
            a = await avanzar(Verbos(), LEYO, semilla=semilla)(log)
            e = await exigir()(merge(log, a))

            assert await planner(Verbos(), LEYO, semilla=semilla)(log) == merge(a, e)


VENTA = (
    PlanOp("add_step", "diagnostico"),
    PlanOp("add_step", "plaga", padre="diagnostico", completes_when="dijo"),
    PlanOp("add_step", "producto", padre="diagnostico", completes_when="leyo", exige=("buscar", "fuente")),
    PlanOp("add_step", "dosis", padre="diagnostico", exige=("dosis", "fuente")),
)

PASOS = {
    "buscar": PlanOp("add_step", "buscar", "encontrar", completes_when="leyo"),
    "dosis": PlanOp("add_step", "dosis", "traer la dosis"),
    "fuente": PlanOp("add_step", "fuente", "citar"),
}


class TestDerivar:
    async def test_con_la_hoja_activa_sin_exigencia_no_siembra_nada(self) -> None:
        assert await derivar(PASOS)(Log(course=VENTA)) == ZERO

    async def test_siembra_lo_que_exige_la_hoja_activa_y_lo_dibuja(self) -> None:
        seen = Log(course=(*VENTA, PlanOp("mark", "plaga", status="done")))

        out = await derivar(PASOS)(seen)

        assert out.steps == (PASOS["buscar"], PASOS["fuente"])
        assert out.said[0].text.startswith(ESTADO) and "buscar" in out.said[0].text

    async def test_no_vuelve_a_sembrar_lo_que_ya_esta_ni_lo_saltado(self) -> None:
        seen = Log(
            course=(*VENTA, PlanOp("mark", "plaga", status="done")),
            steps=(PASOS["buscar"], PASOS["fuente"], PlanOp("skip", "fuente")),
        )

        assert await derivar(PASOS)(seen) == ZERO

    async def test_cuando_la_hoja_cambia_a_mitad_del_turno_la_checklist_crece(self) -> None:
        """producto cerró con el grep: ahora el turno exige la dosis, en este turno."""
        seen = Log(
            course=(*VENTA, PlanOp("mark", "plaga", status="done"), PlanOp("mark", "producto", status="done")),
            steps=(PASOS["buscar"], PASOS["fuente"]),
        )

        out = await derivar(PASOS)(seen)

        assert out.steps == (PASOS["dosis"],)

    async def test_un_id_fuera_del_catalogo_es_un_fail(self) -> None:
        raro = (PlanOp("add_step", "x", exige=("nadie",)),)

        out = await derivar(PASOS)(Log(course=raro))

        assert out.steps == () and "nadie" in out.fails[0].reason


class TestElPasoDeLaVenta:
    """Las cuatro células juntas, un paso a la vez, sin modelo."""

    def paso(self, v: Verbos) -> object:
        registro: Registro = {"leyo": lambda log: log.reads > 0, "dijo": lambda log: "mosca" in log.said[0].text}
        return then(
            avanzar(v, registro, canal="course", semilla=VENTA, etiqueta="venta"),
            derivar(PASOS),
            avanzar(v, registro),
            exigir(),
        )

    async def test_turno_uno_pregunta_sin_checklist(self) -> None:
        seen = Log(said=(dicho("tengo una plaga", Role.USER), dicho("¿qué plaga ves?")))

        out = await self.paso(Verbos())(seen)  # type: ignore[operator]

        assert out.vote is Status.QUIET
        assert out.steps == ()
        assert proyectar(out.course)[1].status == "todo"

    async def test_turno_dos_cierra_plaga_y_exige_buscar(self) -> None:
        seen = Log(
            said=(dicho("mosca blanca", Role.USER), dicho("Metaveria sirve. ¿cuántas manzanas?")),
            course=VENTA,
        )

        out = await self.paso(Verbos())(seen)  # type: ignore[operator]

        assert proyectar(merge(seen, out).course)[1].status == "done"
        assert [op.id for op in out.steps] == ["buscar", "fuente"]
        assert out.vote is Status.CONTINUE and FALTAN in out.said[-1].text

    async def test_el_fold_entre_turnos_no_recierra(self) -> None:
        previos = (*VENTA, PlanOp("mark", "plaga", status="done"))
        seen = Log(said=(dicho("dos", Role.USER), dicho(bloque("x=1"))), course=previos, reads=1)

        out = await self.paso(Verbos())(seen)  # type: ignore[operator]

        assert [op.id for op in out.course] == ["producto"]
        assert "plaga" not in [op.id for op in out.course]


class TestRenderConEtapas:
    def test_sangra_las_hojas_y_cierra_la_etapa_por_sus_hijos(self) -> None:
        plan = proyectar((*VENTA, PlanOp("mark", "plaga", status="done"), PlanOp("skip", "producto"), PlanOp("skip", "dosis")))

        texto = render_plan(plan, "[venta] estado:")

        assert texto.splitlines()[0] == "[venta] estado:"
        assert texto.splitlines()[1].startswith("[x] diagnostico")
        assert texto.splitlines()[2].startswith("    [x] plaga")
