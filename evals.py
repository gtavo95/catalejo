"""Catalejo contra un corpus de verdad, con las respuestas ya escritas por alguien.

    uv run evals.py                 corre las 20 preguntas de la wiki
    uv run evals.py <caso> <caso>   corre un subconjunto por id
    uv run evals.py <caso>          corre una y muestra la transcripción
    uv run evals.py --recurse       las 20, con `llm` disponible
    uv run evals.py --contenedor    las 20, con el REPL en un proceso hijo que se mata si tarda
    uv run evals.py --agro --contenedor  las agronómicas, con el REPL en un proceso hijo
    uv run evals.py --un-paso       las 20, con el grep de antes que muestra líneas, sin `read`
    uv run evals.py --agro --un-paso     las agronómicas, en un paso (el baseline de `dos_pasos`)
    uv run evals.py --repeats=3     cada caso tres veces, para ver cuál flipa
    uv run evals.py --agro          las preguntas agronómicas, contra el agente de agro.py
    uv run evals.py --agro --plan   las mismas, con la checklist prendida
    uv run evals.py --agro --sin-citas   las mismas, sin la célula de citas (el arm baseline)
    uv run evals.py --agro --sin-ontologia   las mismas, sin `objetivos` en el REPL (el baseline de `ontologia`)
    uv run evals.py --agro --sesion          las mismas, con el plan de la venta sembrado en cada pregunta
    uv run evals.py --agro --flujos          las conversaciones de `flujos.tsv`, turno a turno, sin plan
    uv run evals.py --agro --flujos --sesion las mismas, con el plan de la venta entre turnos (el arm de `plan_de_venta`)
    uv run evals.py --openai        el mismo examen contra OpenAI

El corpus es `successo-okf`, la wiki de producto de una empresa de bioinsumos:
102 archivos markdown, 604 KB, ~150k tokens. Y acá está la diferencia con los
otros demos: esto SÍ entra en un prompt. Gemini se lo come entero.

Por eso es la prueba que faltaba. Hasta ahora el argumento del RLM era "el
contexto no entra"; con este corpus el argumento tiene que ser otro, y es el
precio: meterlo entero cuesta ~150k tokens por turno, y consultarlo con grep
cuesta lo que ocupe la línea que buscabas.

`evals/preguntas.tsv` trae 20 preguntas de cliente con la página de la que tiene
que salir la respuesta y el criterio de qué NO hacer. No las escribí yo y no las
escribió el modelo: son las que alguien vio fallar atendiendo. Varias son trampas
puras, que es lo que las hace valer:

  un producto descontinuado, sin filas de precio. Invita a inventar uno.
  un producto con la sección vacía. Invita a traer la de otro producto.
  un producto sin página. Invita a rellenar.
  un producto que no está en el catálogo. Invita a ofrecer un sustituto.

Las cuatro son el mismo modo de falla que la corrida contra v1: el modelo no
encuentra y contesta igual. Un grep que devuelve "0 líneas casan" es un hecho
duro, y esa es la ventaja que un prompt relleno no tiene: ahí la ausencia hay que
notarla entre 150k tokens de ruido.

Los ids de los casos viven en el TSV y no acá, porque nombran productos de un
cliente. Correr el script con un id que no existe los lista todos.

El corpus EXCLUYE `evals/`, que trae la respuesta de cada pregunta escrita al
lado. Meterlo sería hacerse trampa al solitario.

# Dos montajes

Correr el examen y contestarlo son dos trabajos. Este archivo hace el primero:
lee el TSV, corre los casos, califica la cita y reporta. Quién contesta es un
`Montaje`, y hay dos. El de la wiki se arma más abajo. El de `--agro` es el
agente de `agro.py` tal cual, con su corpus de dos carpetas, su CONTRATO y su
índice de productos en el REPL.

Esa separación es todo el punto del eval agronómico: la baja de 108k a 10.4k
tokens se midió sobre `agro.py`, así que confirmarla pide correr `agro.py`, no
un primo suyo con otro prompt.

# Repetir

`--repeats=N` corre cada caso N veces, porque n=1 es suerte y este stack lo tiene
medido en `bitacora.tsv`. n=3 es la primera lectura y n=6 confirma un default.
Lo que n compra es varianza adentro de un caso, así que lo que hay que mirar del
reporte es la lista de flippers: el caso estable repetido seis veces es gasto
muerto.

Los tokens tienen piso de ruido, y es alto. El mismo código, con luna, corrido
tres veces el mismo día dio 9.547, 12.576 y 11.294 medios por corrida (filas
`cabecera_por_documento` y `cita_cerrada` de la bitácora), con la célula de
citas sin hablar nunca en las dos últimas. Un Δ de tokens menor que ±30% a n=3
no se lee; lo que sí se lee es aciertos, flippers, y los avisos de células.

Va con igual y no con espacio, y no es gusto: el filtro por id se lleva todo
argumento que no empiece con guion, así que `--repeats 3` leería `3` como id de
caso y correría cero casos.

# Flujos

Una pregunta suelta no mide una venta. La venta es "tengo una plaga en el
tomate", "mosca blanca", "dos manzanas": tres turnos donde el primero no tiene
respuesta sino pregunta, el segundo recién puede buscar, y el tercero depende de
que el agente recuerde el producto del segundo. `flujos.tsv` trae esas
conversaciones, una fila por turno, con lo que cada turno tiene que citar. Y un
valor nuevo en `paginas`, `pregunta`, para el turno cuya respuesta correcta es
preguntarle algo al cliente: se califica por el signo de pregunta, que es un
piso y se dice.

Un flujo corre como corre `agro.py`: un workspace por conversación, la historia
en el primer mensaje, y con `--sesion` el canal `course` sembrado al principio y
arrastrado turno a turno. Lo que sale es un `Corrida` por turno, con el id
`flujo:n`, así que `resumir` los cuenta igual que a las preguntas sueltas y el
reporte agrega, por flujo, cuántas conversaciones acertaron todos sus turnos.

La historia que viaja es la que vio el cliente (`agro.final`), no la cruda: si
un turno salió como "no pude revisar el catálogo", el siguiente pregunta sobre
eso. Lo que se califica sigue siendo el texto crudo.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import agro
from catalejo.core import Cell, Log, Message, PlanOp, Role, proyectar
from catalejo.llm import Provider
from catalejo.rlm import Contenedor, Handle, Repl, Workspace, drive, inventada, recurse, rutas

AVISO = re.compile(r"\[(\w+)\]")

BUNDLE = Path(__file__).resolve().parent.parent / "okf" / "successo-okf"

CITA = (
    "\n\nAl final de tu respuesta agrega una línea `FUENTE: ruta/al/archivo.md` con el "
    "archivo del que sacaste el dato, o `FUENTE: ninguna` si el bundle no lo tiene."
)


PREGUNTA = "pregunta"
"""En `paginas`: la respuesta correcta de este turno es preguntarle algo al cliente."""


@dataclass(frozen=True, slots=True)
class Caso:
    id: str
    pregunta: str
    paginas: tuple[str, ...]
    seccion: str
    criterio: str


def esperadas(paginas: str) -> tuple[str, ...]:
    """La columna `paginas` como tupla: vacía es `ninguna`, `pregunta` queda tal cual."""
    return () if paginas.strip() == "ninguna" else tuple(paginas.split())


def filas(tsv: str) -> list[list[str]]:
    out = []
    for fila in (BUNDLE / "evals" / tsv).read_text().splitlines():
        if fila.strip() and not fila.startswith("#"):
            out.append(fila.split("\t"))
    return out


def casos(tsv: str) -> list[Caso]:
    out = []
    for id_, pregunta, paginas, seccion, criterio in filas(tsv):
        out.append(Caso(id_, pregunta, esperadas(paginas), seccion, criterio))
    return out


@dataclass(frozen=True, slots=True)
class Flujo:
    """Una conversación: sus turnos en orden, cada uno un `Caso` con id `flujo:n`."""

    id: str
    turnos: tuple[Caso, ...]


def flujos(tsv: str) -> list[Flujo]:
    """`flujos.tsv`: una fila por turno, con el flujo adelante y el número del turno.

    El número está en el archivo para que una fila se lea sola, y se verifica:
    un turno 3 sin turno 2 es un error del TSV, no una conversación corta.
    """
    por_flujo: dict[str, list[Caso]] = {}
    for flujo, turno, pregunta, paginas, seccion, criterio in filas(tsv):
        vistos = por_flujo.setdefault(flujo, [])
        if turno != str(len(vistos) + 1):
            raise SystemExit(f"{tsv}: el flujo {flujo} tiene el turno {turno} después de {len(vistos)}")
        vistos.append(Caso(f"{flujo}:{turno}", pregunta, esperadas(paginas), seccion, criterio))
    return [Flujo(id_, tuple(turnos)) for id_, turnos in por_flujo.items()]


def corpus() -> str:
    """Los markdown del bundle, cada uno con su ruta adelante.

    Sin `evals/`, que trae las respuestas, y sin `.claude/`, que son instrucciones
    para un agente y no contenido de la wiki.
    """
    partes = []
    for f in sorted(BUNDLE.rglob("*.md")):
        rel = f.relative_to(BUNDLE)
        if rel.parts[0] in {"evals", ".claude"}:
            continue
        partes.append(f"=== {rel} ===\n{f.read_text(errors='replace')}")
    return "\n\n".join(partes)


ESQUEMA = (
    "la wiki de producto de una empresa de bioinsumos agrícolas, un archivo markdown por "
    "página, cada uno precedido por una línea `=== ruta ===`. En `productos/` hay una "
    "página por producto con frontmatter (tags, cultivos certificados, estado), resumen, "
    "presentaciones con precios, ingredientes, equivalencias de nombre por país y la ficha "
    "técnica del fabricante. En `company_info/` están contacto, horarios, pagos, cuentas "
    "bancarias y envíos. En `ontologia/` los vocabularios cerrados. En `raw/md/` el OCR "
    "crudo de las fichas, sin curar"
)


def wiki(
    texto: str,
    modelo: Provider,
    *,
    recursivo: bool,
    contenedor: bool = False,
    dos_pasos: bool = True,
) -> tuple[Cell, Repl]:
    """El agente de la wiki: el REPL sobre el bundle entero y el contrato de cita.

    `contenedor` corre el código del modelo en un proceso hijo. Lo que el modelo
    ve no cambia, porque `var` y `tools` son los mismos; lo único que puede
    aparecer distinto es un `[repl]` diciendo que el snippet se mató por tardar.

    `dos_pasos` sí cambia lo que ve, y va por default: `grep` sin `doc=` dice solo
    dónde, y `read` trae la página entera. `--un-paso` lo apaga, y es el baseline
    de la fila `dos_pasos` de la bitácora.
    """
    ws: Repl
    if recursivo:
        ws = recurse(
            texto,
            modelo,
            var="wiki",
            depth=0,
            budget=60_000,
            contenedor=contenedor,
            dos_pasos=dos_pasos,
        )
    elif contenedor:
        ws = Contenedor(texto, var="wiki", dos_pasos=dos_pasos)
    else:
        ws = Workspace(texto, var="wiki", dos_pasos=dos_pasos)
    h = Handle(
        var="wiki",
        schema=ESQUEMA,
        size=f"{len(texto) // 1000} KB, ~{len(texto) // 4000}k tokens",
        tools=ws.tools,
    )
    return drive(modelo, h, ws, keep_recent=6, max_steps=12, budget=120_000), ws


Course = tuple[PlanOp, ...]
Turno = Callable[[Cell, Repl, str, agro.Historia, Course | None], Awaitable[Log]]
"""Un turno: el agente, su workspace, la pregunta, lo hablado y la venta hasta acá."""


@dataclass(frozen=True, slots=True)
class Montaje:
    """Quién contesta el examen, sobre qué corpus y con qué preguntas.

    Existe para que el eval agronómico corra el agente de `agro.py` y no una
    imitación. Un montaje trae el TSV, el corpus, cómo se arma el agente y cómo
    se corre un turno, que es lo único que cambia entre los dos: el de la wiki
    pega el contrato de cita y llama al agente; el de agro es `agro.turno`, con
    la historia y el canal `course` adentro, que es lo que hace que un flujo
    corra igual que una sesión en la terminal.

    `sesion` dice si el turno arranca con la venta sembrada (`course=()`) o sin
    venta (`None`). Lo sabe el montaje y no el que corre, porque es el mismo
    dato con el que se armó el agente.
    """

    tsv: str
    corpus: Callable[[], str]
    montar: Callable[[str, Provider], tuple[Cell, Repl]]
    turno: Turno
    sesion: bool = False


def montaje(argv: list[str]) -> Montaje:
    """El montaje que pidieron. Con `--agro`, el de `agro.py` tal cual.

    `--plan` es la palanca del A/B de la checklist y solo tiene sentido con
    `--agro`, porque la semilla y el registro de compuertas son del dominio
    agronómico. El montaje de la wiki lo ignora, y eso es correcto: el plan no es
    una mejora del motor que se prenda en todas partes, es un cableado de un
    agente.

    `--sesion` es la otra palanca de `agro.py`, y la misma regla: la semilla de
    la venta es del dominio. Con `--flujos` el TSV es el de las conversaciones,
    y solo existe para agro, porque una venta es del asesor y no de la wiki.

    `--contenedor` es lo contrario: el mismo arm con otro motor abajo, y entra
    en los dos montajes. `--un-paso` también entra en los dos, y ese sí es otro
    arm: cambia lo que `grep` devuelve y lo que la nota de herramientas dice.
    """
    contenedor = "--contenedor" in argv
    dos_pasos = "--un-paso" not in argv
    if "--agro" in argv:
        plan = "--plan" in argv
        sesion = "--sesion" in argv
        if plan and sesion:
            raise SystemExit("--plan y --sesion no van juntos: comparten `steps` y el colector")
        citas = "--sin-citas" not in argv
        ontologia = "--sin-ontologia" not in argv
        return Montaje(
            tsv="flujos.tsv" if "--flujos" in argv else "agro.tsv",
            corpus=agro.corpus,
            montar=lambda texto, modelo: agro.armar(
                texto,
                modelo,
                ver=False,
                plan=plan,
                sesion=sesion,
                citas=citas,
                ontologia=ontologia,
                contenedor=contenedor,
                dos_pasos=dos_pasos,
            ),
            turno=lambda agente, ws, pregunta, historia, course: agro.turno(
                agente, ws, pregunta, historia, plan=plan, course=course, ontologia=ontologia
            ),
            sesion=sesion,
        )
    if "--flujos" in argv:
        raise SystemExit("--flujos es del montaje de agro: agregá --agro")
    recursivo = "--recurse" in argv

    async def turno(agente: Cell, ws: Repl, pregunta: str, historia: agro.Historia, course: Course | None) -> Log:
        return await agente(Log(said=(Message(Role.USER, pregunta + CITA),)))

    return Montaje(
        tsv="preguntas.tsv",
        corpus=corpus,
        montar=lambda texto, modelo: wiki(
            texto, modelo, recursivo=recursivo, contenedor=contenedor, dos_pasos=dos_pasos
        ),
        turno=turno,
    )


def repeticiones(argv: list[str]) -> int:
    for a in argv:
        if a.startswith("--repeats="):
            valor = a.split("=", 1)[1]
            if not valor.isdigit() or int(valor) < 1:
                raise SystemExit(f"--repeats={valor}: tiene que ser un entero de 1 para arriba")
            return int(valor)
    return 1


async def correr(caso: Caso, texto: str, modelo: Provider, m: Montaje) -> tuple[Log, int, float]:
    """Una corrida de un caso, con su workspace recién armado y cerrado al final.

    El workspace se arma acá adentro y no afuera a propósito. Las variables
    persisten entre corridas, que es lo que se quiere en una sesión y lo que
    arruina una medición: el índice que el modelo construya en el caso 1 le
    quedaría servido al caso 2 y el segundo saldría barato por el trabajo del
    primero.

    Y se cierra acá porque nadie más lo tiene: `Corrida` no guarda el workspace,
    y un proceso hijo no muere cuando se lo deja de referenciar. `daemon=True`
    solo cubre la salida del padre; sin el `cerrar`, `--repeats=3` deja sesenta
    intérpretes vivos, cada uno con el corpus adentro, hasta que termine la
    corrida entera. Del workspace sobrevive un número, las lecturas delegadas.
    """
    agente, ws = m.montar(texto, modelo)
    t0 = time.monotonic()
    try:
        out = await m.turno(agente, ws, caso.pregunta, (), () if m.sesion else None)
    finally:
        ws.cerrar()
    delegadas = ws.bridge.calls if ws.bridge is not None else 0
    return out, delegadas, time.monotonic() - t0


def final(out: Log) -> str:
    """El último dicho en prosa, CRUDO.

    A propósito no hace lo que `agro.final`, que reemplaza una respuesta sin
    fundamento por un "no sé". Acá el texto tal como salió es el dato: es lo que
    se imprime bajo `contestó:` cuando un caso falla, y es lo que dejó ver que el
    modelo estaba deliberando en vez de contestar. Un corredor de exámenes que
    limpia lo que mide no sirve para diagnosticar.
    """
    for m in reversed(out.said):
        if m.role is Role.ASSISTANT and "```" not in m.text:
            return m.text.strip()
    return "(sin respuesta en prosa)"


def acierta(caso: Caso, texto: str) -> bool:
    """Si citó la página que el eval espera.

    Con `ninguna` el acierto es decir que no hay: se acepta la cita explícita o
    que no haya citado ninguna página de producto inventada.

    Con `pregunta` el acierto es haber preguntado, y se lee por el signo. Es un
    piso: una respuesta que recomienda tres productos y cierra con "¿cuál
    querés?" pasa, y el criterio de la fila es el que dice si eso estuvo bien.
    """
    bajo = texto.lower()
    if caso.paginas == (PREGUNTA,):
        return "?" in texto
    if not caso.paginas:
        return "ninguna" in bajo
    return any(p.lower() in bajo for p in caso.paginas)


@dataclass(frozen=True, slots=True)
class Corrida:
    """Lo que dejó una corrida de un caso, ya calificada.

    Es data pura y no guarda el workspace: de ahí solo sobrevive `delegadas`.
    Así la aritmética del reporte se puede testear sin red ni bundle.
    """

    caso: Caso
    vuelta: int
    out: Log
    respuesta: str
    seg: float
    ok: bool
    fantasmas: tuple[str, ...] = ()
    delegadas: int = 0
    venta: str = ""
    """Con `--sesion`, dónde quedó la venta después de este turno (`agro.estado_venta`)."""

    @property
    def perdida(self) -> bool:
        """El proveedor no contestó, así que esta muestra no dice nada de la calidad.

        Se reconoce por `Fail.who == "model"`, que es el turno vacío. Los fallos
        de `loop` (se quedó sin pasos o sin presupuesto) y de `grounding`
        (contestó sin mirar) NO son muestras perdidas: son resultados.
        """
        return any(f.who == "model" for f in self.out.fails)


@dataclass(frozen=True, slots=True)
class Resumen:
    n: int
    casos: int
    estables: tuple[str, ...]
    flippers: tuple[str, ...]
    caidos: tuple[str, ...]
    mudos: tuple[str, ...]
    fantasmas: dict[str, tuple[str, ...]]
    aciertos: int
    vivas: int
    perdidas: int
    spent: int
    gastado: int
    turnos: float
    delegadas: int
    reads: int
    avisos: dict[str, int]


def avisos(corridas: Sequence[Corrida]) -> dict[str, int]:
    """Cuántas veces habló cada célula, por su prefijo: `[grounding]`, `[cita]`.

    Sin esto un A/B de una célula no se puede leer: si el arm con la célula
    gasta más, hay que saber si es porque avisó y el modelo volvió a contestar
    o porque el proveedor tuvo un día caro. `[repl]` es la salida del
    executor y no un aviso, así que no cuenta.
    """
    cuenta: dict[str, int] = {}
    for c in corridas:
        for m in c.out.said:
            m_ = AVISO.match(m.text) if m.role is Role.USER else None
            if m_ and m_.group(1) != "repl":
                cuenta[m_.group(1)] = cuenta.get(m_.group(1), 0) + 1
    return cuenta


def resumir(corridas: Sequence[Corrida], n: int) -> Resumen:
    """La aritmética del reporte, aparte para poder testearla sin gastar un peso.

    Tres reglas viven acá y no adentro de un `print`.

    Una corrida que murió por un turno vacío del proveedor es una MUESTRA
    PERDIDA, no un fallo de calidad. Contarla como fallo sesga la comparación a
    favor del arm que tuvo suerte con la API, que es lo contrario de lo que uno
    quiere medir. Sale del denominador y se reporta aparte.

    `estable` es el caso que acertó en TODAS sus corridas vivas. A n=1 coincide
    con lo que se reportaba antes de que existieran las repeticiones, así que las
    filas viejas de la bitácora siguen siendo comparables.

    `flipper` es el que acertó en algunas y no en otras. Es lo único que la
    escalera de n compra, y es la lista de qué profundizar a n=6. Un caso estable
    repetido seis veces no informa nada.

    El gasto medio va sobre las corridas vivas, que son las que se comparan entre
    arms. El total facturado va aparte y entero, porque una muestra perdida se
    paga igual.
    """
    por_caso: dict[str, list[Corrida]] = {}
    for c in corridas:
        por_caso.setdefault(c.caso.id, []).append(c)

    estables, flippers, caidos, mudos = [], [], [], []
    fantasmas: dict[str, tuple[str, ...]] = {}
    for cid, grupo in por_caso.items():
        vivas = [c for c in grupo if not c.perdida]
        ok = sum(1 for c in vivas if c.ok)
        if not vivas:
            mudos.append(cid)
        elif ok == len(vivas):
            estables.append(cid)
        elif ok == 0:
            caidos.append(cid)
        else:
            flippers.append(cid)
        malas = tuple(sorted({r for c in grupo for r in c.fantasmas}))
        if malas:
            fantasmas[cid] = malas

    vivas = [c for c in corridas if not c.perdida]
    piso = max(len(vivas), 1)
    return Resumen(
        n=n,
        casos=len(por_caso),
        estables=tuple(estables),
        flippers=tuple(flippers),
        caidos=tuple(caidos),
        mudos=tuple(mudos),
        fantasmas=fantasmas,
        aciertos=sum(1 for c in vivas if c.ok),
        vivas=len(vivas),
        perdidas=len(corridas) - len(vivas),
        spent=sum(c.out.spent for c in vivas) // piso,
        gastado=sum(c.out.spent for c in corridas),
        turnos=sum(len(c.out.said) for c in vivas) / piso,
        delegadas=sum(c.delegadas for c in corridas),
        reads=sum(c.out.reads for c in corridas),
        avisos=avisos(corridas),
    )


def calificar(
    caso: Caso,
    vuelta: int,
    out: Log,
    seg: float,
    delegadas: int,
    reales: frozenset[str],
    venta: str = "",
) -> Corrida:
    """El `Corrida` de un turno: la respuesta cruda, si acertó y qué inventó."""
    respuesta = final(out)
    return Corrida(
        caso=caso,
        vuelta=vuelta,
        out=out,
        respuesta=respuesta,
        seg=seg,
        ok=acierta(caso, respuesta),
        fantasmas=inventada(respuesta, reales),
        delegadas=delegadas,
        venta=venta,
    )


async def correr_flujo(
    flujo: Flujo, vuelta: int, texto: str, modelo: Provider, m: Montaje, reales: frozenset[str]
) -> list[Corrida]:
    """Una conversación entera, como la corre `agro.main`: un workspace, la
    historia de a tres, y el canal `course` arrastrado si el montaje lleva venta.

    Cada turno es un `Corrida` propio, con lo que ese turno gastó. Las lecturas
    delegadas del puente son acumuladas, así que se resta lo que ya había.
    """
    agente, ws = m.montar(texto, modelo)
    historia: agro.Historia = ()
    course: Course | None = () if m.sesion else None
    hechas: list[Corrida] = []
    leidas = 0
    try:
        for caso in flujo.turnos:
            t0 = time.monotonic()
            out = await m.turno(agente, ws, caso.pregunta, historia, course)
            seg = time.monotonic() - t0
            calls = ws.bridge.calls if ws.bridge is not None else 0
            venta = ""
            if course is not None:
                course = (*course, *out.course)
                venta = agro.estado_venta(proyectar(course))
            hechas.append(calificar(caso, vuelta, out, seg, calls - leidas, reales, venta))
            leidas = calls
            vista = agro.final(out, fundar=course is None or bool(out.steps))
            historia = (*historia, (caso.pregunta, vista))[-3:]
    finally:
        ws.cerrar()
    return hechas


def enteros(corridas: Sequence[Corrida]) -> dict[str, tuple[int, int]]:
    """Por flujo: conversaciones que acertaron TODOS sus turnos, sobre las vivas.

    Una conversación es (flujo, vuelta). Está viva si ningún turno perdió la
    muestra, y entera si todos acertaron. Es la cuenta que a un vendedor le
    importa: una venta con el turno 2 bien y el 3 mal no cerró.
    """
    por_charla: dict[tuple[str, int], list[Corrida]] = {}
    for c in corridas:
        flujo = c.caso.id.rsplit(":", 1)[0]
        por_charla.setdefault((flujo, c.vuelta), []).append(c)
    cuenta: dict[str, list[int]] = {}
    for (flujo, _), turnos in por_charla.items():
        viva = not any(c.perdida for c in turnos)
        entera = viva and all(c.ok for c in turnos)
        acumulado = cuenta.setdefault(flujo, [0, 0])
        acumulado[0] += entera
        acumulado[1] += viva
    return {flujo: (a, b) for flujo, (a, b) in cuenta.items()}


def sha() -> str:
    """El commit de esta corrida, con `-dirty` si el árbol no estaba limpio."""
    try:
        cabeza = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        sucio = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return f"{cabeza}-dirty" if sucio else cabeza


def fila(r: Resumen) -> str:
    """La fila de `bitacora.tsv` ya armada, para pegar.

    El sha va resuelto porque la cabecera del archivo lo pide y transcribirlo a
    mano se presta a error. `palanca`, `arm` y `nota` salen en `·`: eso lo sabe
    el que corrió, no el script. Y sin la fila el experimento se evapora en
    stdout, que es la única forma segura de perder una medición que ya se pagó.
    """
    return "\t".join(
        [
            date.today().isoformat(),
            sha(),
            "·",
            "·",
            str(r.n),
            str(r.spent),
            f"{r.turnos:.1f}",
            f"{len(r.estables)}/{r.casos}",
            str(r.perdidas),
            "·",
        ]
    )


def linea(c: Corrida, n: int) -> str:
    """Una corrida en una línea del reporte, como va saliendo."""
    marca = "···" if c.perdida else ("ok " if c.ok else "MAL")
    cual = f" #{c.vuelta + 1}" if n > 1 else ""
    venta = f"  {c.venta}" if c.venta else ""
    return (
        f"  {marca} {c.caso.id:24s}{cual} {c.out.spent:>7,} tok  "
        f"{c.seg:5.1f}s  {len(c.out.said):2d} turnos{venta}"
    )


def transcripcion(out: Log) -> None:
    for msg in out.said:
        quien = {Role.USER: "repl", Role.ASSISTANT: "modelo"}.get(msg.role, msg.role.value)
        print(f"\n\033[1m[{quien}]\033[0m {msg.text.strip()[:2000]}")


async def main(argv: list[str]) -> None:
    n = repeticiones(argv)
    m = montaje(argv)
    pedidos = [a for a in argv if not a.startswith("-")]
    charlas = m.tsv == "flujos.tsv"
    todos: Sequence[Caso | Flujo] = flujos(m.tsv) if charlas else casos(m.tsv)
    elegidos = [c for c in todos if c.id in pedidos] if pedidos else list(todos)
    if not elegidos:
        que = "flujo" if charlas else "caso"
        print(f"no hay {que} {pedidos}. Hay: {', '.join(c.id for c in todos)}")
        return
    corridas: list[Caso] = [
        t for e in elegidos for t in (e.turnos if isinstance(e, Flujo) else (e,))
    ]

    texto = m.corpus()
    reales = rutas(texto)
    modelo = agro.proveedor(argv)
    cuantos = f"{len(elegidos)} flujos, {len(corridas)} turnos" if charlas else f"{len(corridas)} casos"
    print(
        f"corpus: {len(texto):,} caracteres (~{len(texto) // 4:,} tokens), "
        f"{cuantos}, n={n} · {modelo.model}{' · con plan de venta' if m.sesion else ''}"
    )
    sem = asyncio.Semaphore(4)
    solo = len(elegidos) == 1 and n == 1

    async def uno(caso: Caso, vuelta: int) -> list[Corrida]:
        async with sem:
            out, delegadas, seg = await correr(caso, texto, modelo, m)
        c = calificar(caso, vuelta, out, seg, delegadas, reales)
        print(linea(c, n))
        if solo:
            transcripcion(out)
        return [c]

    async def charla(flujo: Flujo, vuelta: int) -> list[Corrida]:
        async with sem:
            turnos = await correr_flujo(flujo, vuelta, texto, modelo, m, reales)
        for c in turnos:
            print(linea(c, n))
            if solo:
                transcripcion(c.out)
        return turnos

    try:
        grupos = await asyncio.gather(
            *(
                (charla(e, i) if isinstance(e, Flujo) else uno(e, i))
                for e in elegidos
                for i in range(n)
            )
        )
    finally:
        await modelo.aclose()
    hechas = [c for grupo in grupos for c in grupo]
    orden = {c.id: i for i, c in enumerate(corridas)}
    filas = sorted(hechas, key=lambda c: (orden[c.caso.id], c.vuelta))
    r = resumir(filas, n)

    print("\n\033[1m[resultado]\033[0m")
    titulo = "citó la página correcta" if n == 1 else f"acertó las {n}"
    print(f"  {titulo:25s} {len(r.estables)}/{r.casos} casos")
    if r.flippers:
        print(f"  flipan                    {len(r.flippers)}  {', '.join(r.flippers)}")
    if r.caidos:
        print(f"  fallan siempre            {len(r.caidos)}  {', '.join(r.caidos)}")
    if r.mudos:
        print(f"  sin una muestra viva      {len(r.mudos)}  {', '.join(r.mudos)}")
    print(f"  aciertos                  {r.aciertos}/{r.vivas} corridas vivas")
    if r.perdidas:
        print(f"  muestras perdidas         {r.perdidas}  (turno vacío, fuera del denominador)")
    print(f"  citó una ruta inexistente {len(r.fantasmas)}/{r.casos}  {r.fantasmas or ''}")
    if charlas:
        for flujo, (ent, vivas) in enteros(filas).items():
            gasto = sum(c.out.spent for c in filas if c.caso.id.startswith(f"{flujo}:")) // max(vivas, 1)
            print(f"  {flujo:26s}{ent}/{vivas} conversaciones enteras, {gasto:,} tok por conversación")
    print(f"  tokens                    {r.spent:,} medios por corrida, {r.gastado:,} facturados")
    if r.delegadas:
        print(f"  lecturas delegadas        {r.delegadas}")
    print(f"  consultas al corpus       {r.reads}")
    if r.avisos:
        avisado = ", ".join(f"[{quien}] {n}" for quien, n in sorted(r.avisos.items()))
        print(f"  avisos de células         {avisado}")
    print(f"  relleno equivalente       {len(texto) // 4 * max(r.vivas, 1):,} tokens")
    print(f"\n  para bitacora.tsv, con palanca, arm y nota tuyas:\n  {fila(r)}")

    for caso in corridas:
        grupo = [c for c in filas if c.caso.id == caso.id]
        if all(c.ok and not c.out.fails and not c.fantasmas for c in grupo):
            continue
        print(f"\n  \033[1m{caso.id}\033[0m  esperaba {caso.paginas or ('ninguna',)}")
        print(f"    criterio: {caso.criterio}")
        malas = [c for c in grupo if not c.ok] or grupo
        print(f"    contestó: {malas[0].respuesta[:600]}")
        for c in grupo:
            if c.out.fails:
                print(f"    #{c.vuelta + 1} fails: {[f.reason for f in c.out.fails]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
