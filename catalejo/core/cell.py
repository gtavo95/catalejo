"""La célula y las formas de componerlas.

La asociatividad de la composición no se programa acá: se hereda del monoide del
Log. Eso es lo que compra tener una sola operación.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from functools import reduce
from typing import Protocol

from .log import ZERO, Fail, Log, Rule, Status, drop, merge


class Cell(Protocol):
    """Lee todo lo dicho hasta ahora y devuelve lo que agrega. Solo lo que agrega.

    No muta nada, así que quien la compuso decide si el agregado aterriza. Eso es
    lo que hace posible que otra célula lo revise antes.

    Dos ausencias a propósito. No hay un argumento aparte para el mensaje que
    entra, porque un turno del usuario es un Log con un Message adentro y no
    necesita tipo propio. Y no levanta excepciones, porque una falla es el canal
    `fails`: tenerla adentro del Log es lo que hace la composición total, y así
    ningún combinador puede cortar un turno a la mitad sin nada que mostrar.

    Es `__call__`, así que cualquier función async ya es una célula.
    """

    async def __call__(self, seen: Log) -> Log: ...


def then(*cells: Cell) -> Cell:
    """Corre las células en orden y devuelve todo lo que agregaron.

    Cada una ve lo que llegó más lo que agregaron las anteriores:

        (f ▷ g)(seen) = f(seen) • g(seen • f(seen))

    Pasar `seen` hacia adentro es lo que hace `then` asociativo. Escribir `acc`
    solo se ve casi igual y rompe la ley en silencio, porque un `then` anidado le
    tapa el contexto a sus propias células. Un test que solo mire el resultado no
    lo agarra: hay que mirar qué vio cada una.

    `then()` es `identity()`, y `then(c)` se comporta como `c`.
    """

    async def chained(seen: Log) -> Log:
        acc = ZERO
        for cell in cells:
            acc = merge(acc, await cell(merge(seen, acc)))
        return acc

    return chained


def identity() -> Cell:
    """No agrega nada: es el neutro de `then`.

    Sirve donde una célula está apagada, y así la forma de la composición deja de
    depender de la configuración.
    """

    async def nothing(seen: Log) -> Log:
        return ZERO

    return nothing


def border(cell: Cell, rule: Rule) -> Cell:
    """Le aplica la regla a lo que la célula devuelve, una sola vez, al salir.

    Qué cruza el borde de un alcance es una decisión, y la de por defecto no es
    "todo". En el repo ya hay tres bordes distintos y hasta ahora cada uno estaba
    escrito a mano. `then` y `fanout` no tienen: cruza todo. `loop` reemplaza el
    voto por el del último paso. Un sub-agente deja pasar `spent` como número y
    colapsa todo lo que dijo en un solo texto.

    Ninguno de los dos que ya existen se puede escribir con esto, y las razones
    valen la pena. El del `loop` necesita el voto del último paso, y eso no está
    en el Log acumulado: la suma por máximo ya se lo comió. El del sub-agente
    sale del tipo, porque devuelve un `str` que entra al REPL como salida y el
    gasto cruza por `Metered` en vez de por el Log. Este combinador es para el
    tercer caso, que es el borde que sí se queda adentro del álgebra.

    Las reglas que son homomorfismos, como `drop`, se pueden aplicar adentro o
    afuera de una acumulación sin que nada cambie. Las que colapsan, no, y por eso
    existe este combinador: acá arriba corren exactamente una vez.

    Cuidado con dónde se lo pone. `border(loop(c, ...), r)` recorta lo que sale
    del loop; `loop(border(c, r), ...)` recorta cada paso, y ahí lo recortado no
    lo ve el paso siguiente.
    """

    async def bounded(seen: Log) -> Log:
        return rule(await cell(seen))

    return bounded


def mute(cell: Cell) -> Cell:
    """Corre la célula y le tira lo que dijo. Lo que gastó y lo que falló queda.

    Es para la que trabaja y no habla: un crítico que vota pero no ensucia la
    transcripción, un sub-agente cuyo ida y vuelta interno no le importa al padre.
    Callarla no la abarata, y por eso `spent` sigue cruzando.
    """
    return border(cell, drop("said"))


def quiet(cell: Cell) -> Cell:
    """Corre la célula y le saca el voto. Todo lo demás queda.

    Es para la que mira y no manda. Sin esto, cualquier célula que vote CONTINUE
    tiene al loop de rehén, porque el voto se junta por máximo. Con esto se puede
    cablear una observadora sin darle poder sobre la terminación.
    """
    return border(cell, drop("vote"))


def only(pred: Callable[[Log], bool], cell: Cell) -> Cell:
    """Corre `cell` cuando `pred` se cumple, y no agrega nada si no."""

    async def guarded(seen: Log) -> Log:
        return await cell(seen) if pred(seen) else ZERO

    return guarded


def fanout(*cells: Cell) -> Cell:
    """Corre las células al mismo tiempo. Cada una ve exactamente lo que llegó.

    Ninguna ve a las otras, que es lo que tiene que significar "al mismo tiempo"
    si el resultado va a ser reproducible.

    Acá se cobra la conmutatividad. Como merge es conmutativo en `fails` y
    `vote`, en esos canales el orden de las células no se puede observar. `said`
    es una lista y concatenar no es conmutativo, así que las que hablan sí
    dependen de su posición, y por eso junta en el orden en que se las dio y no
    en el que terminaron. La regla práctica sale del álgebra y no del gusto:
    búsquedas y juicios en paralelo, las que escriben prosa en secuencia.
    """

    async def parallel(seen: Log) -> Log:
        added = await asyncio.gather(*(cell(seen) for cell in cells))
        return reduce(merge, added, ZERO)

    return parallel


def retry(cell: Cell, *, attempts: int) -> Cell:
    """Repite la célula mientras traiga `fails`, y tira lo que dijo el intento perdido.

    Es el primer combinador que descarta una propuesta. Puede hacerlo porque una
    célula devuelve lo que agrega y no muta nada, así que quien la compuso decide
    si el agregado aterriza. Ese permiso estaba en el diseño desde el principio y
    hasta ahora no lo usaba nadie.

    Reintenta por `fails` y no por el voto, porque `fails` es el canal que
    significa que se rompió la maquinaria. Un snippet que revienta no es un
    `Fail`, es flujo normal del REPL, así que esto no se dispara con eso, que es
    justo lo que se quiere: el modelo lee el error y lo corrige solo.

    Del intento perdido sobreviven `spent` y `fails`. El gasto porque la llamada
    caída se paga igual, y la falla porque es verdad que pasó y el que llame
    tiene derecho a verla. Se van `said`, `vote`, `reads`, `steps` y `course`. Los dos primeros son
    obvios: media respuesta no se muestra y un voto de algo que no aterrizó no
    manda. `reads` es el que importa: si quedara, un intento que leyó el contexto
    y después reventó le dejaría el piso servido a `grounded`, y el intento que
    sí aterrizó podría contestar de memoria pareciendo fundado. Los dos planes
    se van por lo mismo: un intento que se tiró no movió nada.

    Un intento que vota HALT no se reintenta. Alguien decidió que no hay otra
    vuelta, y eso vale acá igual que en `loop`.

    El último intento vuelve entero, con lo que dijo y con las fallas de todos.
    Fallar hasta el final no es abortar: el que llamó se lleva lo que haya y
    decide, que es la misma regla que hace total a toda la composición.

    `attempts=1` es correr la célula una vez, y `attempts=0` no agrega nada,
    igual que `then()`.
    """
    perdido = drop("said", "vote", "reads", "steps", "course")

    async def retried(seen: Log) -> Log:
        acc = ZERO
        for intento in range(attempts):
            added = await cell(merge(seen, acc))
            ultimo = intento == attempts - 1
            if not added.fails or ultimo or added.vote is Status.HALT:
                return merge(acc, added)
            acc = merge(acc, perdido(added))
        return acc

    return retried


def loop(cell: Cell, *, max_steps: int, budget: int | None = None) -> Cell:
    """Repite la célula mientras el último paso pida seguir.

    `cell` es la que se repite. Cada vuelta ve lo que llegó más lo que acumularon
    las vueltas anteriores, así que un paso puede leer lo que dijo el anterior.
    `max_steps` es vueltas; `budget` es tokens, medidos en `spent`, y `None` lo
    apaga.

    Mira `added.vote`, el voto de ESTE paso, y no `acc.vote`, el voto acumulado.
    La diferencia no es un detalle: `vote` se junta por máximo, así que una vez
    que alguien votó CONTINUE el acumulado se queda en CONTINUE para siempre y un
    loop que lo mirara no terminaría nunca. El acumulado dice "alguien pidió
    seguir alguna vez"; el paso dice "todavía falta".

    Un paso que vota QUIET corta. Nadie pidió otra vuelta, así que un cableado a
    medio hacer se detiene en vez de quemar tokens. Uno que vota HALT también, y
    ese es el punto de que HALT exista: sin él CONTINUE le gana a todo y ninguna
    célula puede frenar el turno siguiente.

    El HALT sale hacia afuera, porque lo que cruza el borde es el voto del último
    paso. Un sub-loop que frenó le reporta HALT al padre y el padre también
    frena. Es lo correcto: el "seguí pensando" de adentro no tiene por qué
    escaparse, pero "esto no sigue" sí.

    `max_steps` y `budget` son los topes mecánicos que garantizan que esto
    termina, incluso con un modelo que siga pidiendo más. Todo el resto de la
    política vive en las células, que la expresan votando.

    Quedarse sin pasos o sin presupuesto no levanta una excepción: agrega un
    `Fail`, porque es un hecho sobre el turno como cualquier otro. Y el voto que
    sale es CONTINUE, que es lo honesto: lo cortaron a la mitad y seguía
    queriendo trabajar.

    El voto de salida es el del último paso, no el máximo acumulado, y eso es lo
    que hace que un loop sirva de célula. El máximo vale adentro de un alcance;
    cruzar el borde de un sub-loop resume, no acumula. Sin eso, un sub-loop que
    dio dos vueltas le reporta CONTINUE al padre para siempre y el padre no para
    nunca. Dicho de otro modo: el "seguí pensando" de adentro no se le escapa al
    de afuera, que es exactamente lo que tiene que pasar cuando un modelo manda
    un sub-modelo a leer por él.

    Sigue siendo una célula, así que un loop entero cabe adentro de otro. Ahí va
    a vivir la recursión de RLM.
    """

    async def looped(seen: Log) -> Log:
        acc = ZERO
        for _ in range(max_steps):
            added = await cell(merge(seen, acc))
            acc = merge(acc, added)
            if added.vote is not Status.CONTINUE:
                return replace(acc, vote=added.vote)
            if budget is not None and acc.spent >= budget:
                agotado = f"presupuesto agotado: {acc.spent}/{budget}"
                return merge(acc, Log(fails=(Fail("loop", agotado),)))
        return merge(acc, Log(fails=(Fail("loop", f"tope de pasos: {max_steps}"),)))

    return looped
