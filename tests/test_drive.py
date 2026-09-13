"""El cableado con nombre, y el turno entero con la checklist prendida.

`then(worker, executor, grounded)` adentro de un `loop` estaba escrito a mano en
`agro.py` y en `evals.py`, así que el eval medía una copia del agente. Esto prueba
la copia única, y de paso lo que pasa cuando se le agrega una célula al final.
"""

from catalejo.core import ZERO, Cell, Log, Message, PlanOp, Role, Status, proyectar
from catalejo.llm import Stub
from catalejo.rlm import Handle, Registro, Verbos, Workspace, drive, planner
from catalejo.rlm.celulas.grounding import AVISO
from catalejo.rlm.celulas.planner import FALTAN, INCOMPLETO

PAYLOAD = "manual\ngarantia: 24 meses\nfin"

LEYO: Registro = {"leyo": lambda log: log.reads > 0}

SEMILLA = (
    PlanOp("add_step", "buscar", "encontrar el dato", completes_when="leyo"),
    PlanOp("add_step", "decir", "contestar con la unidad"),
)


def bloque(code: str) -> str:
    return f"```python\n{code}\n```"


def pregunta(texto: str) -> Log:
    return Log(said=(Message(Role.USER, texto),))


def armar(model: Stub, ws: Workspace, *extras: Cell) -> Cell:
    return drive(model, Handle(tools=ws.tools), ws, max_steps=6, extras=extras)


class TestElCableado:
    async def test_consulta_y_contesta(self) -> None:
        model = Stub(bloque("print(grep(ctx, 'garantia'))"), "la garantía es de 24 meses")
        ws = Workspace(PAYLOAD)

        out = await armar(model, ws)(pregunta("cuánta garantía"))

        assert out.said[-1].text == "la garantía es de 24 meses"
        assert out.vote is Status.DONE
        assert out.fails == ()
        assert out.reads == 1

    async def test_trae_grounded_puesto(self) -> None:
        """Es la razón de que exista: el cableado de la política viaja con el
        cableado, así que nadie puede armar el agente sin el juez."""
        model = Stub("son 24 meses", bloque("print(grep(ctx, 'garantia'))"), "24 meses")
        ws = Workspace(PAYLOAD)

        out = await armar(model, ws)(pregunta("cuánta garantía"))

        assert Message(Role.USER, AVISO.format(var="ctx")) in out.said

    async def test_los_extras_ven_el_paso_entero(self) -> None:
        visto: list[int] = []

        async def mirona(seen: Log) -> Log:
            visto.append(len(seen.said))
            return ZERO

        model = Stub(bloque("print(1)"), "listo")
        ws = Workspace(PAYLOAD)

        await armar(model, ws, mirona)(pregunta("hola"))

        assert visto == [3, 4]

    async def test_el_voto_que_sale_es_el_del_ultimo_paso(self) -> None:
        model = Stub(bloque("print(1)"), bloque("print(2)"))
        ws = Workspace(PAYLOAD)

        out = await drive(model, Handle(tools=ws.tools), ws, max_steps=2)(pregunta("hola"))

        assert out.vote is Status.CONTINUE
        assert out.fails and "tope de pasos" in out.fails[0].reason


class TestElTurnoConPlan:
    """El turno entero, offline. Es la prueba de que la terminación cambió de
    dueño: antes la decidía la forma del mensaje, ahora la forma del mensaje más
    un plan que no tiene pasos abiertos."""

    async def test_el_plan_manda_a_cerrar_antes_de_contestar(self) -> None:
        v = Verbos()
        ws = Workspace(PAYLOAD, extra=v.builtins)
        model = Stub(
            bloque('mark("buscar", "active")\nprint(grep(ctx, "garantia"))'),
            "la garantía es de 24 meses",
            bloque('mark("decir", "done")\nprint("cerrado")'),
            "la garantía es de 24 meses",
        )

        out = await armar(model, ws, planner(v, LEYO, semilla=SEMILLA))(pregunta("garantía"))

        plan = proyectar(out.steps)
        assert [s.status for s in plan] == ["done", "done"]
        assert out.vote is Status.DONE
        assert out.fails == ()
        assert any(FALTAN in m.text for m in out.said)
        assert out.said[-1].text == "la garantía es de 24 meses"

    async def test_la_compuerta_la_cierra_el_hecho_y_no_el_modelo(self) -> None:
        """`buscar` nunca se marca `done` en el guion: lo cierra `leyo` cuando el
        REPL devuelve algo."""
        v = Verbos()
        ws = Workspace(PAYLOAD, extra=v.builtins)
        model = Stub(
            bloque('print(grep(ctx, "garantia"))\nmark("decir", "done")'),
            "24 meses",
        )

        out = await armar(model, ws, planner(v, LEYO, semilla=SEMILLA))(pregunta("garantía"))

        assert proyectar(out.steps)[0].status == "done"
        assert out.vote is Status.DONE
        assert out.fails == ()

    async def test_si_ignora_el_aviso_la_respuesta_sale_etiquetada(self) -> None:
        """La misma decisión que `grounded`: la respuesta sale, pero sale con el
        `Fail`. Insistir hasta el tope son seis llamadas para la misma conclusión."""
        v = Verbos()
        ws = Workspace(PAYLOAD, extra=v.builtins)
        model = Stub(
            bloque('print(grep(ctx, "garantia"))'),
            "24 meses",
            "en serio, 24 meses",
        )

        out = await armar(model, ws, planner(v, LEYO, semilla=SEMILLA))(pregunta("garantía"))

        assert INCOMPLETO in out.fails
        assert out.vote is Status.DONE
        assert proyectar(out.steps)[1].status == "todo"
