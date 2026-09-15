"""Un solo tipo de dato y una sola operación.

El estado del agente y la propuesta de una célula tienen la misma forma. No son
tipos distintos, es el mismo tipo mirado con distinto alcance, y por eso una sola
`merge` sirve adentro de un paso y entre pasos.

Los errores viven adentro del Log, en `fails`. Un ejecutor que reventó es un
hecho sobre el turno, igual que algo que se dijo. Por eso ninguna función de este
módulo levanta una excepción.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Literal

from .message import Conversation
from .plan import PlanOp


@dataclass(frozen=True, slots=True, order=True)
class Fail:
    """Algo que no funcionó. Un hecho, no una razón para abandonar el turno.

    Una célula lo reporta y quien la compuso decide si es fatal.
    """

    who: str
    reason: str


class Status(IntEnum):
    """El voto sobre si el turno terminó.

    Está ordenado, y juntar dos votos es tomar el mayor: si alguien dice que
    falta información, falta información, sin importar quién votó último. Ese
    orden es lo que hace el canal seguro de juntar en cualquier orden, que es lo
    que después deja paralelizar.

    HALT es el único absorbente del álgebra, y está acá porque sin él CONTINUE
    manda y ninguna célula puede decir "pará". Es el voto de un juez que ve algo
    que no se arregla con otra vuelta: una inyección, una respuesta que no puede
    salir. No reintroduce el error como cortocircuito, que es lo que este diseño
    descartó, porque frena el turno siguiente y no el que está corriendo. Las
    células ya cableadas de este paso corren igual y lo que dijeron y lo que
    falló aterriza igual: los otros canales no se enteran de que alguien votó
    HALT.
    """

    QUIET = 0
    DONE = 1
    CONTINUE = 2
    HALT = 3


@dataclass(frozen=True, slots=True)
class Log:
    """Todo lo dicho, en canales. Cada canal trae su propia álgebra.

    said   una lista. El orden es el significado, así que solo concatena.
    steps  una lista. Los ops que movieron el plan del turno, en orden.
    course una lista. Los ops que movieron el plan de la conversación, en orden.
    fails  un conjunto. El mismo error reportado dos veces es un error.
    vote   el máximo del orden de Status.
    spent  una suma. Los tokens que costó todo esto.
    reads  una suma. Cuántas veces el REPL devolvió algo.

    `steps` no es el plan, son las MOVIDAS del plan. El plan se calcula
    plegándolas con `proyectar`, así que no hay una copia que pueda quedar
    desincronizada y cada célula ve el plan tal como estaba cuando corrió. Trae
    un tipo propio por la misma razón que `said` trae `Message`: el álgebra ya
    declara la forma de lo que viaja.

    `course` es el mismo tipo con otro alcance: las movidas del plan que dura la
    conversación entera, la venta con sus etapas. Dos canales del mismo tipo que
    se distinguen solo por lo que significan, como `spent` y `reads` son dos
    enteros que se suman. Es el único canal que el host siembra: el Log arranca
    limpio en `reads`, `said` y `fails` en cada turno, que es lo que `grounded`
    necesita, y `course` entra con la historia de la venta y sale con lo que este
    turno le agregó. Por la ley del fold, guardar eso al final de lo anterior da
    el mismo plan que si todo hubiera pasado en un solo turno.

    `reads` es el canal que dice si la respuesta tiene de dónde salir. En cero, el
    modelo no consultó nada y lo que conteste sale de su memoria, no del contexto.
    Es un piso, no una prueba: cuenta que el REPL devolvió algo, no que eso que
    devolvió tenga que ver con lo que después se afirma.
    """

    said: Conversation = ()
    steps: tuple[PlanOp, ...] = ()
    course: tuple[PlanOp, ...] = ()
    fails: tuple[Fail, ...] = ()
    vote: Status = Status.QUIET
    spent: int = 0
    reads: int = 0


ZERO = Log()


def merge(a: Log, b: Log) -> Log:
    """La única operación. Asociativa, con ZERO como neutro.

    En todo menos `said` es además conmutativa. Idempotente lo es solo en `fails` y
    `vote`: `said` es una lista y `spent` y `reads` son sumas, y las tres cosas
    dicen la verdad. Repetir una respuesta la duplica, reintentar cuesta plata, y
    consultar dos veces son dos consultas.

    Esa frontera es exactamente la que dice qué se puede correr en paralelo y qué
    se puede reintentar gratis.

    Agregar un canal es agregar una línea acá, no un `if` en el engine.
    """
    return Log(
        said=a.said + b.said,
        steps=a.steps + b.steps,
        course=a.course + b.course,
        fails=tuple(sorted(set(a.fails) | set(b.fails))),
        vote=max(a.vote, b.vote),
        spent=a.spent + b.spent,
        reads=a.reads + b.reads,
    )


def normal(log: Log) -> Log:
    """Lleva un Log escrito a mano a la forma canónica que produce merge.

    Las leyes valen entre logs normales: la idempotencia necesita la entrada ya
    deduplicada y la igualdad necesita un orden fijo. Juntar con ZERO es cómo se
    llega.
    """
    return merge(ZERO, log)


type Rule = Callable[[Log], Log]

type Channel = Literal["said", "steps", "course", "fails", "vote", "spent", "reads"]


def drop(*channels: Channel) -> Rule:
    """Vacía canales, dejando el resto como estaba.

    El vacío de cada canal lo saca de ZERO, así que agregar un canal sigue
    siendo una línea en `merge` y ninguna acá.

    Es un homomorfismo de monoide: vaciar la suma es lo mismo que sumar los
    vaciados, `h(a • b) == h(a) • h(b)`. Por eso se puede aplicar adentro o
    afuera de una acumulación sin que el resultado cambie, que es lo que la
    hace segura de mover de lugar. Una regla que colapse un canal en vez de
    vaciarlo, como quedarse con el último dicho, no lo es, y esas van una sola
    vez en el borde y nunca adentro de una acumulación.

    `drop("spent")` es la que hay que mirar dos veces: le esconde el gasto al
    presupuesto del padre y ahí el árbol se financia solo. Vale cuando ese gasto
    ya se contó en otro lado, como el que `Metered` mide en la fuente, y no vale
    para que las cuentas den lindas.
    """
    vacios = {canal: getattr(ZERO, canal) for canal in channels}

    def dropped(log: Log) -> Log:
        return replace(log, **vacios)

    return dropped
