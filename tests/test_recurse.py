import asyncio
import os

import pytest

from catalejo.core import ZERO, Conversation, Fail, Log, Message, Role, Status, loop, then
from catalejo.llm import Gemini, Reply
from catalejo.repl import Bridge, Handle, answer, executor, note, recurse, worker

COSTO = 10


class Router:
    """Un modelo que contesta según lo que le preguntaron, no según el orden.

    Un guion por posición se vuelve ilegible acá: en una corrida recursiva se
    mezclan los turnos del padre, los del hijo y las lecturas planas.
    """

    def __init__(self, *reglas: tuple[str, str], sino: str = "no sé") -> None:
        self.reglas = reglas
        self.sino = sino
        self.visto: list[Conversation] = []

    async def complete(self, conv: Conversation) -> Reply:
        self.visto.append(conv)
        ultimo = conv[-1].text
        for clave, respuesta in self.reglas:
            if clave in ultimo:
                return Reply(Message(Role.ASSISTANT, respuesta), spent=COSTO)
        return Reply(Message(Role.ASSISTANT, self.sino), spent=COSTO)


def bloque(code: str) -> str:
    return f"```python\n{code}\n```"


class TestAnswer:
    def test_devuelve_el_ultimo_dicho_en_prosa(self) -> None:
        out = Log(
            said=(
                Message(Role.USER, "pregunta"),
                Message(Role.ASSISTANT, bloque("print(1)")),
                Message(Role.USER, "[repl] salida:\n1"),
                Message(Role.ASSISTANT, "la respuesta es 1"),
            )
        )

        assert answer(out) == "la respuesta es 1"

    def test_saltea_el_codigo(self) -> None:
        """Un sub-agente sin pasos tiene código como último dicho, y devolver eso
        sería devolver la mitad de un pensamiento."""
        out = Log(
            said=(
                Message(Role.ASSISTANT, "voy viendo"),
                Message(Role.ASSISTANT, bloque("print(2)")),
            )
        )

        assert answer(out) == "voy viendo"

    def test_avisa_cuando_se_corto(self) -> None:
        """Una respuesta incompleta que no se anuncia es el mismo error que un
        grep que recorta en silencio."""
        out = Log(
            said=(Message(Role.ASSISTANT, "encontré tres"),),
            fails=(Fail("loop", "tope de pasos: 6"),),
        )

        salida = answer(out)

        assert salida.startswith("encontré tres")
        assert "tope de pasos: 6" in salida
        assert "incompleta" in salida

    def test_sin_dichos_lo_dice(self) -> None:
        assert "no llegó a contestar" in answer(ZERO)


class TestNote:
    def test_dice_cuando_NO_delegar(self) -> None:
        """Un modelo con un builtin nuevo lo usa para todo, y pagar tokens para
        contar líneas cuesta plata y encima se equivoca."""
        assert "Contar, filtrar y ordenar hazlo tú con Python" in note(1)

    def test_muestra_el_barrido_en_paralelo(self) -> None:
        assert "trozos = [ctx[i:i+40000] for i in range(0, len(ctx), 40000)]" in note(1)

    def test_en_el_fondo_no_nombra_rlm(self) -> None:
        """Un builtin que no está tampoco se nombra: nombrarlo quema un turno."""
        assert "rlm(pregunta" not in note(0)
        assert "llm(pregunta" in note(0)


class TestLlm:
    async def test_el_modelo_puede_leer_un_pedazo(self) -> None:
        model = Router(("de qué habla", "de tostadoras"))
        ws = recurse("el manual de la tostadora", model)

        out = await ws.run("print(llm('de qué habla esto', ctx))")

        assert out.stdout == "de tostadoras\n"
        assert out.err == ""

    async def test_una_lista_devuelve_una_lista(self) -> None:
        model = Router(("uno", "primero"), ("dos", "segundo"))
        ws = recurse("", model)

        out = await ws.run("print(llm('cuál es', ['uno', 'dos']))")

        assert out.stdout == "['primero', 'segundo']\n"

    async def test_la_lista_corre_en_paralelo(self) -> None:
        """El punto entero del barrido por pedazos: veinte trozos son un round
        trip, no veinte."""
        arrancaron = 0
        picos: list[int] = []

        class Lento:
            async def complete(self, conv: Conversation) -> Reply:
                nonlocal arrancaron
                arrancaron += 1
                picos.append(arrancaron)
                await asyncio.sleep(0.01)
                arrancaron -= 1
                return Reply(Message(Role.ASSISTANT, "ok"), spent=COSTO)

        ws = recurse("", Lento(), paralelo=4)

        await ws.run("print(len(llm('x', ['a', 'b', 'c', 'd'])))")

        assert max(picos) == 4

    async def test_el_paralelo_tiene_tope(self) -> None:
        vivos = 0
        picos: list[int] = []

        class Lento:
            async def complete(self, conv: Conversation) -> Reply:
                nonlocal vivos
                vivos += 1
                picos.append(vivos)
                await asyncio.sleep(0.01)
                vivos -= 1
                return Reply(Message(Role.ASSISTANT, "ok"), spent=COSTO)

        ws = recurse("", Lento(), paralelo=2)

        await ws.run("print(len(llm('x', ['a', 'b', 'c', 'd', 'e'])))")

        assert max(picos) == 2

    async def test_una_llamada_que_falla_no_mata_al_batch(self) -> None:
        """Levantar acá dejaría a las otras respuestas del batch sin dueño."""

        class Roto:
            def __init__(self) -> None:
                self.n = 0

            async def complete(self, conv: Conversation) -> Reply:
                self.n += 1
                if self.n == 1:
                    raise TimeoutError("504")
                return Reply(Message(Role.ASSISTANT, "bien"), spent=COSTO)

        ws = recurse("", Roto(), paralelo=1)

        out = await ws.run("print(llm('x', ['a', 'b']))")

        assert "TimeoutError: 504" in out.stdout
        assert "bien" in out.stdout
        assert out.err == ""


class TestElGasto:
    async def test_sube_a_la_salida_de_la_corrida(self) -> None:
        ws = recurse("", Router(), budget=1000)

        out = await ws.run("llm('x', ['a', 'b', 'c'])")

        assert out.spent == 3 * COSTO

    async def test_correr_python_no_cuesta(self) -> None:
        ws = recurse("hola", Router())

        assert (await ws.run("print(len(ctx))")).spent == 0

    async def test_cruza_el_borde_hasta_el_log(self) -> None:
        """Sin esto el gasto de los hijos es invisible para el budget del padre."""
        model = Router()
        ws = recurse("", model)

        out = await executor(ws)(Log(said=(Message(Role.ASSISTANT, bloque("llm('x', 'y')")),)))

        assert out.spent == COSTO
        assert out.vote is Status.CONTINUE

    async def test_el_presupuesto_agotado_no_llama(self) -> None:
        model = Router()
        ws = recurse("", model, budget=COSTO)

        out = await ws.run("print(llm('x', ['a', 'b', 'c']))")

        assert len(model.visto) == 1  # la primera pasa, las otras ven la caja vacía
        assert out.stdout.count("sin presupuesto para delegar") == 2


class TestRlm:
    async def test_el_sub_agente_tiene_su_propio_contexto(self) -> None:
        """El hijo ve la rebanada, no el contexto del padre."""
        model = Router(("cuánto mide", bloque("print(len(ctx))")), ("[repl]", "mide 5"))
        ws = recurse("un contexto largo de verdad", model)

        out = await ws.run("print(rlm('cuánto mide', ctx[:5]))")

        assert out.stdout == "mide 5\n"
        corrido = "\n".join(m.text for conv in model.visto for m in conv)
        assert "un contexto largo de verdad" not in corrido

    async def test_en_el_fondo_no_hay_rlm(self) -> None:
        """La recursión termina leyendo, no cortándose en seco."""
        ws = recurse("", Router(), depth=0)

        out = await ws.run("rlm('x', 'y')")

        assert "NameError" in out.err
        assert (await ws.run("print(llm('x', 'y'))")).err == ""

    async def test_el_hijo_puede_delegar_a_su_vez(self) -> None:
        model = Router(
            ("nivel dos", bloque("print(llm('leé esto', ctx))")),
            ("leé esto", "es un pedazo"),
            ("[repl]", "el nieto dijo que es un pedazo"),
        )
        ws = recurse("payload", model, depth=1)

        out = await ws.run("print(rlm('nivel dos', 'la rebanada'))")

        assert out.stdout == "el nieto dijo que es un pedazo\n"
        assert out.spent == 3 * COSTO


class TestNadaSeCuentaDosVeces:
    async def test_el_total_es_exactamente_lo_que_se_llamo(self) -> None:
        """La trampa del diseño: el hijo mide su propio gasto Y el padre mide la
        corrida entera, y las dos ventanas se solapan en el tiempo."""
        model = Router(
            ("cuántos hay", bloque("print(rlm('mirá esto', ctx[:4]))")),
            ("mirá esto", bloque("print(llm('y esto', ctx))")),
            ("y esto", "cuatro"),
            ("[repl] salida:\ncuatro", "el nieto dijo cuatro"),
            ("[repl] salida:\nel nieto", "son cuatro"),
        )
        ws = recurse("un payload cualquiera", model)
        handle = Handle(tools=ws.tools)

        agente = loop(then(worker(model, handle), executor(ws)), max_steps=8)
        out = await agente(Log(said=(Message(Role.USER, "cuántos hay"),)))

        assert out.said[-1].text == "son cuatro"
        assert out.spent == len(model.visto) * COSTO

    async def test_el_budget_del_loop_ve_al_arbol(self) -> None:
        model = Router(
            ("cuántos hay", bloque("print(llm('leé', ctx))")),
            ("leé", "muchos"),
        )
        ws = recurse("payload", model)

        agente = loop(
            then(worker(model, Handle(tools=ws.tools)), executor(ws)), max_steps=8, budget=15
        )
        out = await agente(Log(said=(Message(Role.USER, "cuántos hay"),)))

        # el turno gastó 10 del worker más 10 del hijo: el tope de 15 lo ve
        assert out.spent == 2 * COSTO
        assert out.fails == (Fail("loop", "presupuesto agotado: 20/15"),)


class TestElPuente:
    async def test_sin_loop_no_cruza_nadie(self) -> None:
        """El builtin corre en el hilo del exec, y ese hilo no alcanza el loop por
        su cuenta: se lo tiene que dejar el Workspace antes de cruzar."""

        async def algo() -> str:
            return "nunca"

        with pytest.raises(RuntimeError, match="event loop"):
            Bridge().wait(algo())

    async def test_la_nota_del_workspace_llega_al_preambulo(self) -> None:
        """Si el Handle no toma la nota del workspace, el modelo nunca se entera
        de que puede delegar."""
        ws = recurse("", Router())
        model = Router()

        await worker(model, Handle(tools=ws.tools))(Log(said=(Message(Role.USER, "hola"),)))

        preambulo = model.visto[0][0].text
        assert "llm(pregunta, texto)" in preambulo
        assert "grep(texto, patron)" in preambulo


# Prosa escrita a mano, no plantillas: el tema de cada una solo se saca leyendo,
# que es la única situación donde delegar en un modelo tiene sentido.
NOTAS = [
    "El contenedor quedó varado once días. Los estibadores pararon el martes y no "
    "hubo forma de mover nada del muelle hasta que levantaron la medida. Avisamos "
    "al cliente el jueves, cuando ya era obvio que no llegábamos con la fecha.",
    "Retención en la terminal: el despachante cargó mal la partida arancelaria y "
    "el organismo pidió reinspección física de los bultos. Se resolvió pagando la "
    "multa, pero fueron seis días parados esperando el turno de verificación.",
    "El proveedor perdió la nave de armado en un incendio la madrugada del 3. No "
    "tienen stock ni línea alternativa hasta marzo. Estamos buscando reemplazo "
    "pero ningún competidor tiene volumen para cubrir el pedido completo.",
]

CLAVE = "\n".join(
    ["registro de mantenimiento de la planta"]
    + [f"linea {i}: revision de rutina sin novedad" for i in range(120)]
    + ["codigo de acceso al tablero principal: 7741"]
    + [f"linea {i}: revision de rutina sin novedad" for i in range(120, 240)]
)


@pytest.mark.skipif(not os.environ.get("RLM_LIVE"), reason="RLM_LIVE=1 para pegarle a la API")
class TestEnVivo:
    async def test_llm_lee_los_pedazos_en_paralelo(self) -> None:
        model = Gemini(thinking="low")
        ws = recurse("\n===\n".join(NOTAS), model, budget=20_000)
        try:
            out = await ws.run(
                "trozos = ctx.split('\\n===\\n')\n"
                "for r in llm('¿por qué se retrasó el envío? Contesta en tres palabras', trozos):\n"
                # Una respuesta con salto de línea desalinea todo lo de abajo, y
                # entonces el test falla por formato en vez de por sentido.
                "    print(' '.join(r.split()))",
            )
        finally:
            await model.aclose()

        assert out.err == ""
        causas = out.stdout.lower().splitlines()
        assert len(causas) == 3
        assert all(c.strip() for c in causas)
        assert not any("la llamada falló" in c for c in causas)

        # Lo que este test tiene que probar es que cada respuesta salió de SU
        # trozo, no qué sinónimo eligió el modelo. La causa del medio se dice de
        # seis maneras distintas ("partida arancelaria", "reinspección física",
        # "error de partida") y todas están bien; fijar una lista de palabras
        # hace que el test falle por vocabulario y no por sentido. Así que se
        # afirma lo que no cambia: la primera habla del paro, la última del
        # incendio, y la del medio no habla de ninguna de las dos.
        assert any(r in causas[0] for r in ("paro", "estibador", "huelga"))
        assert "incendio" in causas[2]
        assert not any(r in causas[1] for r in ("paro", "estibador", "huelga", "incendio"))
        assert out.spent > 0

    async def test_rlm_abre_un_sub_agente_con_su_propio_repl(self) -> None:
        model = Gemini(thinking="low")
        ws = recurse(CLAVE, model, budget=40_000)
        try:
            out = await ws.run(
                "print(rlm('¿cuál es el código de acceso al tablero?', ctx))",
            )
        finally:
            await model.aclose()

        assert out.err == ""
        assert "7741" in out.stdout
        assert out.spent > 0
