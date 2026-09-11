"""Catalejo contra un corpus de verdad, con las respuestas ya escritas por alguien.

    uv run evals.py                 corre las 20 preguntas de la wiki
    uv run evals.py <caso> <caso>   corre un subconjunto por id
    uv run evals.py <caso>          corre una y muestra la transcripción
    uv run evals.py --recurse       las 20, con `llm` disponible
    uv run evals.py --repeats=3     cada caso tres veces, para ver cuál flipa
    uv run evals.py --agro          las preguntas agronómicas, contra el agente de agro.py
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

Va con igual y no con espacio, y no es gusto: el filtro por id se lleva todo
argumento que no empiece con guion, así que `--repeats 3` leería `3` como id de
caso y correría cero casos.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import agro
from catalejo.core import Cell, Log, Message, Role, loop, then
from catalejo.llm import Provider
from catalejo.repl import Handle, Workspace, executor, grounded, recurse, worker

BUNDLE = Path(__file__).resolve().parent.parent / "okf" / "successo-okf"

CABECERA = re.compile(r"^=== (.+) ===$")

CITA = (
    "\n\nAl final de tu respuesta agrega una línea `FUENTE: ruta/al/archivo.md` con el "
    "archivo del que sacaste el dato, o `FUENTE: ninguna` si el bundle no lo tiene."
)


@dataclass(frozen=True, slots=True)
class Caso:
    id: str
    pregunta: str
    paginas: tuple[str, ...]
    seccion: str
    criterio: str


def casos(tsv: str) -> list[Caso]:
    filas = (BUNDLE / "evals" / tsv).read_text().splitlines()
    out = []
    for fila in filas:
        if not fila.strip() or fila.startswith("#"):
            continue
        id_, pregunta, paginas, seccion, criterio = fila.split("\t")
        esperadas = () if paginas.strip() == "ninguna" else tuple(paginas.split())
        out.append(Caso(id_, pregunta, esperadas, seccion, criterio))
    return out


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


def wiki(texto: str, modelo: Provider, *, recursivo: bool) -> tuple[Cell, Workspace]:
    """El agente de la wiki: el REPL sobre el bundle entero y el contrato de cita."""
    ws = (
        recurse(texto, modelo, var="wiki", depth=0, budget=60_000)
        if recursivo
        else Workspace(texto, var="wiki")
    )
    h = Handle(
        var="wiki",
        schema=ESQUEMA,
        size=f"{len(texto) // 1000} KB, ~{len(texto) // 4000}k tokens",
        tools=ws.tools,
    )
    paso = then(worker(modelo, h, keep_recent=6), executor(ws), grounded(ws.var))
    return loop(paso, max_steps=12, budget=120_000), ws


@dataclass(frozen=True, slots=True)
class Montaje:
    """Quién contesta el examen, sobre qué corpus y con qué preguntas.

    Existe para que el eval agronómico corra el agente de `agro.py` y no una
    imitación. Un montaje trae el TSV, el corpus, cómo se arma el agente y cómo
    se redacta el pedido, que es lo único que cambia entre los dos.
    """

    tsv: str
    corpus: Callable[[], str]
    montar: Callable[[str, Provider], tuple[Cell, Workspace]]
    pedir: Callable[[str], str]


def montaje(argv: list[str]) -> Montaje:
    """El montaje que pidieron. Con `--agro`, el de `agro.py` tal cual."""
    if "--agro" in argv:
        return Montaje(
            tsv="agro.tsv",
            corpus=agro.corpus,
            montar=lambda texto, modelo: agro.armar(texto, modelo, ver=False),
            pedir=lambda pregunta: agro.pedido(pregunta, ()),
        )
    recursivo = "--recurse" in argv
    return Montaje(
        tsv="preguntas.tsv",
        corpus=corpus,
        montar=lambda texto, modelo: wiki(texto, modelo, recursivo=recursivo),
        pedir=lambda pregunta: pregunta + CITA,
    )


def repeticiones(argv: list[str]) -> int:
    for a in argv:
        if a.startswith("--repeats="):
            valor = a.split("=", 1)[1]
            if not valor.isdigit() or int(valor) < 1:
                raise SystemExit(f"--repeats={valor}: tiene que ser un entero de 1 para arriba")
            return int(valor)
    return 1


async def correr(
    caso: Caso, texto: str, modelo: Provider, m: Montaje
) -> tuple[Log, Workspace, float]:
    """Una corrida de un caso, con su workspace recién armado.

    El workspace se arma acá adentro y no afuera a propósito. Las variables
    persisten entre corridas, que es lo que se quiere en una sesión y lo que
    arruina una medición: el índice que el modelo construya en el caso 1 le
    quedaría servido al caso 2 y el segundo saldría barato por el trabajo del
    primero.
    """
    agente, ws = m.montar(texto, modelo)
    t0 = time.monotonic()
    out = await agente(Log(said=(Message(Role.USER, m.pedir(caso.pregunta)),)))
    return out, ws, time.monotonic() - t0


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


def rutas(texto: str) -> set[str]:
    """Las rutas que EXISTEN en el corpus que se le dio al modelo.

    Salen de las cabeceras `=== ruta ===` del propio texto y no de un segundo
    recorrido del bundle, así el conjunto es exactamente lo que el modelo pudo
    leer. Importa con `--agro`, donde el corpus son dos carpetas: una cita a un
    archivo que existe en el bundle pero no en el corpus es tan inventada como
    una a un archivo que no existe en ninguna parte.
    """
    return {m.group(1) for m in (CABECERA.match(l) for l in texto.splitlines()) if m}


def citadas(texto: str) -> tuple[str, ...]:
    """Todas las rutas que la respuesta declara como fuente.

    Todas y no la última, porque el CONTRATO de `agro.py` pide una línea
    `FUENTE:` por cada archivo del que salió un dato. Mirando solo la última, una
    ruta fabricada en la primera pasaba sin que nadie la viera, que es justo lo
    que estas dos funciones existen para ver.
    """
    return tuple(
        linea.split(":", 1)[1].strip().strip("`").lstrip("/")
        for linea in texto.splitlines()
        if linea.strip().upper().startswith("FUENTE:")
    )


def inventada(texto: str, reales: set[str]) -> tuple[str, ...]:
    """Las rutas citadas que no existen, si las hay.

    Es el agujero que `reads` no tapa y no hace falta un juez para verlo: el
    modelo SÍ leyó, así que la célula de grounding lo deja pasar, y aun así la
    fuente que declara puede no existir. Una cita es una afirmación sobre un
    conjunto conocido, y eso se verifica con código, que es exacto y gratis.
    """
    return tuple(
        r for r in citadas(texto) if r and r.lower() != "ninguna" and r not in reales
    )


def acierta(caso: Caso, texto: str) -> bool:
    """Si citó la página que el eval espera.

    Con `ninguna` el acierto es decir que no hay: se acepta la cita explícita o
    que no haya citado ninguna página de producto inventada.
    """
    bajo = texto.lower()
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
    )


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


async def main(argv: list[str]) -> None:
    n = repeticiones(argv)
    m = montaje(argv)
    pedidos = [a for a in argv if not a.startswith("-")]
    todos = casos(m.tsv)
    corridas = [c for c in todos if c.id in pedidos] if pedidos else todos
    if not corridas:
        print(f"no hay caso {pedidos}. Hay: {', '.join(c.id for c in todos)}")
        return

    texto = m.corpus()
    reales = rutas(texto)
    modelo = agro.proveedor(argv)
    print(
        f"corpus: {len(texto):,} caracteres (~{len(texto) // 4:,} tokens), "
        f"{len(corridas)} casos, n={n} · {modelo.model}"
    )
    sem = asyncio.Semaphore(4)
    solo = len(corridas) == 1 and n == 1

    async def uno(caso: Caso, vuelta: int) -> Corrida:
        async with sem:
            out, ws, seg = await correr(caso, texto, modelo, m)
        respuesta = final(out)
        c = Corrida(
            caso=caso,
            vuelta=vuelta,
            out=out,
            respuesta=respuesta,
            seg=seg,
            ok=acierta(caso, respuesta),
            fantasmas=inventada(respuesta, reales),
            delegadas=ws.bridge.calls if ws.bridge is not None else 0,
        )
        marca = "···" if c.perdida else ("ok " if c.ok else "MAL")
        cual = f" #{vuelta + 1}" if n > 1 else ""
        print(
            f"  {marca} {caso.id:22s}{cual} {out.spent:>7,} tok  "
            f"{seg:5.1f}s  {len(out.said):2d} turnos"
        )
        if solo:
            for msg in out.said:
                quien = {Role.USER: "repl", Role.ASSISTANT: "modelo"}.get(msg.role, msg.role.value)
                print(f"\n\033[1m[{quien}]\033[0m {msg.text.strip()[:2000]}")
        return c

    try:
        hechas = await asyncio.gather(*(uno(c, i) for c in corridas for i in range(n)))
    finally:
        await modelo.aclose()
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
    print(f"  tokens                    {r.spent:,} medios por corrida, {r.gastado:,} facturados")
    if r.delegadas:
        print(f"  lecturas delegadas        {r.delegadas}")
    print(f"  consultas al corpus       {r.reads}")
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
