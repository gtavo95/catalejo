"""Corridas de verdad contra Gemini sobre un documento que el modelo nunca lee.

    uv run demo.py            consultar con código
    uv run demo.py recurse    consultar con código y delegar la lectura
    uv run demo.py v1         lo mismo sobre el código de v1, que es texto real

El corpus son 60.000 tickets de soporte sintéticos, 8 MB. Meterlo en un prompt
cuesta más de dos millones de tokens por turno y no entra en casi ningún modelo.
Acá el modelo lo consulta escribiendo código y paga los tokens de la
transcripción, que son tres órdenes de magnitud menos.

Las dos preguntas están elegidas para mostrar el reparto del trabajo.

La primera pide contar, fechar y ordenar. Un retriever devuelve los fragmentos
más parecidos a la pregunta, y "cuántos hay" no se parece a nada; el REPL lo
resuelve exacto y gratis.

La segunda pide leer 1768 quejas en texto libre y agruparlas por tema. Eso no lo
hace un regex. El modelo filtra con código, que es la parte barata y exacta, y
delega la lectura de lo filtrado en pedazos que corren en paralelo. Contar con
código, leer con modelos.

Con el corpus sintético la segunda no llega a delegar, y hace bien: diez quejas
distintas repetidas mil veces se dedupean con código y ahí se terminó el trabajo.
Delegar recién paga cuando lo que queda después de filtrar sigue sin entrar en un
prompt, y eso pide entropía de verdad. Por eso está el tercer modo, que apunta al
código Go de v1: 328 archivos, 79 paquetes, 2 MB de texto que nadie generó con
plantillas. Ahí "¿qué hace cada módulo?" no tiene atajo por código.
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

from catalejo.core import Log, Message, Role, loop, then
from catalejo.llm import Gemini
from catalejo.rlm import Handle, Workspace, executor, grounded, recurse, worker

CODIGOS = ["E-102", "E-417", "E-500", "W-31"]
PRODUCTOS = ["tostadora", "licuadora", "cafetera", "batidora"]

QUEJAS = [
    "se apaga sola a los pocos minutos de estar andando",
    "deja de funcionar sin aviso despues de un rato encendida",
    "se corta el funcionamiento y hay que esperar para volver a prenderla",
    "arranca y a los minutos queda muerta hasta que se enfria",
    "el cable se calienta muchisimo cerca de la ficha",
    "sale olor a quemado del enchufe cuando lleva un rato",
    "la ficha quema al tocarla despues de usarla",
    "hace un ruido metalico muy fuerte al arrancar",
    "vibra tanto que se mueve sola sobre la mesada",
    "suena como si algo estuviera suelto adentro",
]

PREGUNTA_REPL = (
    "¿Cuántos tickets mencionan el error E-417, cuál es el más viejo de esos, "
    "y qué producto aparece más entre ellos?"
)
PREGUNTA_RECURSE = (
    "De los tickets con error E-417: agrupá las quejas por tema y decime cuántos "
    "hay de cada tema y cuál domina. Son muchos, así que no los leas todos de una."
)


def corpus(n: int = 60_000, semilla: int = 7) -> str:
    rng = random.Random(semilla)
    rq = random.Random(semilla + 1)
    lineas = []
    for i in range(n):
        dia = rng.randint(1, 28)
        mes = rng.randint(1, 12)
        codigo = rng.choices(CODIGOS, weights=[40, 3, 20, 37])[0]
        producto = rng.choice(PRODUCTOS)
        queja = rq.choices(QUEJAS, weights=[9, 9, 9, 9, 4, 4, 4, 2, 2, 2])[0]
        lineas.append(
            f"ticket {i:06d} | 2024-{mes:02d}-{dia:02d} | producto: {producto} | "
            f"codigo: {codigo} | {queja}"
        )
    return "\n".join(lineas)


V1 = Path(__file__).resolve().parent.parent / "v1"
PREGUNTA_V1 = (
    "Hacé un índice de `v1/planner/internal/plan`: una línea por archivo diciendo qué hace "
    "y si tiene lógica de negocio de verdad o es puro andamiaje. Son 80 archivos y 800 KB "
    "de código: no entran en tu prompt y el encabezado de cada archivo no alcanza para "
    "saber si adentro hay lógica."
)


def corpus_v1() -> str:
    """El código Go de v1, cada archivo con su ruta adelante.

    Sin los `_test.go`, que triplican el bulto sin agregar qué hace cada cosa. La
    línea `=== ruta ===` es la única afordancia que necesita el modelo: con eso
    agrupa por paquete con código, que es exacto y gratis, y delega la lectura.
    """
    partes = []
    for f in sorted(V1.rglob("*.go")):
        if "vendor" in f.parts or f.name.endswith("_test.go"):
            continue
        partes.append(f"=== {f.relative_to(V1.parent)} ===\n{f.read_text(errors='replace')}")
    return "\n".join(partes)


def transcribir(out: Log) -> None:
    for m in out.said:
        quien = {Role.USER: "repl", Role.ASSISTANT: "modelo"}.get(m.role, m.role.value)
        print(f"\n\033[1m[{quien}]\033[0m {m.text.strip()[:1500]}")


def cuenta(out: Log, texto: str, ws: Workspace) -> None:
    print("\n\033[1m[cuenta]\033[0m")
    print(f"  corpus                {len(texto):,} caracteres (~{len(texto) // 4:,} tokens)")
    print(f"  tokens gastados       {out.spent:,}")
    if ws.bridge is not None:
        b = ws.bridge
        cuantas = f"{b.calls} llamadas" if b.calls else "no delegó"
        print(f"  delegado              {cuantas}, {b.spent:,} tokens")
    print(f"  ahorro                {len(texto) // 4 / max(out.spent, 1):.0f}x")
    print(f"  turnos                {len(out.said)}")
    print(f"  voto                  {out.vote.name}")
    if out.fails:
        print(f"  fails                 {out.fails}")


def handle(ws: Workspace, texto: str) -> Handle:
    return Handle(
        var=ws.var,
        schema="una linea por ticket de soporte, con fecha, producto, codigo de error y la queja",
        size=f"{len(texto) // 1_000_000}M caracteres, ~{len(texto) // 4000:,}k tokens",
        tools=ws.tools,
    )


async def main(modo: str) -> None:
    modelo = Gemini(thinking="low")
    if modo == "v1":
        texto = corpus_v1()
        ws = recurse(texto, modelo, var="repo", depth=1, budget=700_000)
        h = Handle(
            var="repo",
            schema="el código Go de un repo, cada archivo precedido por una línea `=== ruta ===`",
            size=f"{len(texto) // 1_000_000}M caracteres, ~{len(texto) // 4000:,}k tokens",
            tools=ws.tools,
        )
        pregunta = PREGUNTA_V1
        techo = 800_000
        pasos = 18
    else:
        texto = corpus()
        recursivo = modo == "recurse"
        ws = (
            recurse(texto, modelo, var="tickets", depth=1, budget=80_000)
            if recursivo
            else Workspace(texto, var="tickets")
        )
        h = handle(ws, texto)
        pregunta = PREGUNTA_RECURSE if recursivo else PREGUNTA_REPL
        techo = 250_000
        pasos = 10

    agente = loop(
        then(worker(modelo, h, keep_recent=6), executor(ws), grounded(ws.var)),
        max_steps=pasos,
        budget=techo,
    )

    try:
        out = await agente(Log(said=(Message(Role.USER, pregunta),)))
    finally:
        await modelo.aclose()

    transcribir(out)
    cuenta(out, texto, ws)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ""))
