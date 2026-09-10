"""Catalejo contra un corpus de verdad, con las respuestas ya escritas por alguien.

    uv run evals.py                 corre las 20 preguntas
    uv run evals.py <caso>          corre una y muestra la transcripción
    uv run evals.py --recurse       las 20, con `llm` disponible

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
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from catalejo.core import Log, Message, Role, loop, then
from catalejo.llm import Gemini, Provider
from catalejo.repl import Handle, Workspace, executor, grounded, recurse, worker

BUNDLE = Path(__file__).resolve().parent.parent / "okf" / "successo-okf"

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


def casos() -> list[Caso]:
    filas = (BUNDLE / "evals" / "preguntas.tsv").read_text().splitlines()
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


async def correr(
    caso: Caso, texto: str, modelo: Provider, *, recursivo: bool
) -> tuple[Log, Workspace, float]:
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
    agente = loop(
        then(worker(modelo, h, keep_recent=6), executor(ws), grounded(ws.var)),
        max_steps=12,
        budget=120_000,
    )
    t0 = time.monotonic()
    out = await agente(Log(said=(Message(Role.USER, caso.pregunta + CITA),)))
    return out, ws, time.monotonic() - t0


def final(out: Log) -> str:
    for m in reversed(out.said):
        if m.role is Role.ASSISTANT and "```" not in m.text:
            return m.text.strip()
    return "(sin respuesta en prosa)"


def rutas() -> set[str]:
    """Las rutas que EXISTEN en el corpus, para poder desmentir una cita."""
    return {
        str(f.relative_to(BUNDLE))
        for f in BUNDLE.rglob("*.md")
        if f.relative_to(BUNDLE).parts[0] not in {"evals", ".claude"}
    }


def citada(texto: str) -> str:
    for linea in reversed(texto.splitlines()):
        if linea.strip().upper().startswith("FUENTE:"):
            return linea.split(":", 1)[1].strip().strip("`").lstrip("/")
    return ""


def inventada(texto: str, reales: set[str]) -> str:
    """La ruta citada que no existe, si la hay.

    Es el agujero que `reads` no tapa y no hace falta un juez para verlo: el
    modelo SÍ leyó, así que la célula de grounding lo deja pasar, y aun así la
    fuente que declara puede no existir. Una cita es una afirmación sobre un
    conjunto conocido, y eso se verifica con código, que es exacto y gratis.
    """
    ruta = citada(texto)
    if not ruta or ruta.lower() == "ninguna":
        return ""
    return "" if ruta in reales else ruta


def acierta(caso: Caso, texto: str) -> bool:
    """Si citó la página que el eval espera.

    Con `ninguna` el acierto es decir que no hay: se acepta la cita explícita o
    que no haya citado ninguna página de producto inventada.
    """
    bajo = texto.lower()
    if not caso.paginas:
        return "ninguna" in bajo
    return any(p.lower() in bajo for p in caso.paginas)


async def main(argv: list[str]) -> None:
    recursivo = "--recurse" in argv
    pedidos = [a for a in argv if not a.startswith("-")]
    todos = casos()
    corridas = [c for c in todos if c.id in pedidos] if pedidos else todos
    if not corridas:
        print(f"no hay caso {pedidos}. Hay: {', '.join(c.id for c in todos)}")
        return

    texto = corpus()
    print(f"corpus: {len(texto):,} caracteres (~{len(texto) // 4:,} tokens), {len(corridas)} casos")
    modelo = Gemini(thinking="low")
    sem = asyncio.Semaphore(4)

    async def uno(caso: Caso) -> tuple[Caso, bool, Log, str, float]:
        async with sem:
            out, _, seg = await correr(caso, texto, modelo, recursivo=recursivo)
        respuesta = final(out)
        ok = acierta(caso, respuesta)
        marca = "ok " if ok else "MAL"
        print(f"  {marca} {caso.id:22s} {out.spent:>7,} tok  {seg:5.1f}s  {len(out.said):2d} turnos")
        if len(corridas) == 1:
            for m in out.said:
                quien = {Role.USER: "repl", Role.ASSISTANT: "modelo"}.get(m.role, m.role.value)
                print(f"\n\033[1m[{quien}]\033[0m {m.text.strip()[:2000]}")
        return caso, ok, out, respuesta, seg

    try:
        hechas = await asyncio.gather(*(uno(c) for c in corridas))
    finally:
        await modelo.aclose()
    filas = sorted(hechas, key=lambda f: [c.id for c in corridas].index(f[0].id))

    print("\n\033[1m[resultado]\033[0m")
    reales = rutas()
    fantasmas = {c.id: r for c, _, _, resp, _ in filas if (r := inventada(resp, reales))}
    bien = sum(1 for _, ok, *_ in filas if ok)
    gasto = sum(o.spent for _, _, o, _, _ in filas)
    print(f"  citó la página correcta  {bien}/{len(filas)}")
    print(f"  citó una ruta inexistente {len(fantasmas)}/{len(filas)}  {fantasmas or ''}")
    print(f"  tokens                   {gasto:,} total, {gasto // max(len(filas), 1):,} por pregunta")
    print(f"  relleno equivalente      {len(texto) // 4 * len(filas):,} tokens")
    for caso, ok, out, respuesta, _ in filas:
        if ok and not out.fails and caso.id not in fantasmas:
            continue
        print(f"\n  \033[1m{caso.id}\033[0m  esperaba {caso.paginas or ('ninguna',)}")
        print(f"    criterio: {caso.criterio}")
        print(f"    contestó: {respuesta[:600]}")
        if out.fails:
            print(f"    fails: {out.fails}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
