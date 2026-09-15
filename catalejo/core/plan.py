"""La checklist como dato, y el intérprete que dice qué significa moverla.

# El plan no se guarda

Lo que viaja en el Log son los OPS. El plan se calcula plegándolos, y de ahí sale
la propiedad que lo hace componible: por la ley del fold,
`fold(s, a ++ b) == fold(fold(s, a), b)`, y como `then` le pasa a cada célula
`merge(seen, acc)`, cada una proyecta el plan tal como estaba cuando corrió sin
que nadie se lo pase. Dos células que tocan el plan en el mismo paso no se pisan.

# Proponer y admitir son dos actos, así que son dos lugares

El modelo propone escribiendo `add_step`, `mark` y `skip` en el REPL. `admitir`
decide qué entra, y `aplicar` es el único árbitro de qué significa cada op. Un op
que no cambia nada no entra, y por eso `aplicar` devuelve el MISMO plan cuando no
aplica: es cómo se detecta, sin una segunda tabla de reglas que se desincronice.

# `done` no está en el vocabulario del modelo

Un paso con compuerta lo cierra su predicado, no el modelo. Si el modelo pudiera
declarar `done`, "el plan está completo" sería otra forma de decir "el modelo dijo
que terminó", que es la terminación que ya teníamos y la razón por la que esto
existe. Un paso SIN compuerta sí lo cierra el modelo, y esa es la válvula: con un
registro de predicados vacío TODOS los pasos caen en la válvula y el plan deja de
garantizar nada. Por eso las compuertas de `agro.py` son predicados sobre el Log y
no campos que no existen.

# Las etapas se calculan, no se guardan

Un plan con dos niveles (la venta: diagnosticar, cobrar, facturar; y adentro de
cada una sus partes) es el mismo plan plano con un campo `padre` en cada paso,
igual que `objetivos` de agro tiene `padre` para las plagas. `aplicar` sigue
siendo tres verbos y `proyectar` sigue siendo un reduce. Lo que sale de ahí son
funciones: una etapa está cerrada cuando todos sus hijos lo están, la etapa
activa es la primera abierta, y la hoja activa es su primer hijo abierto. Nadie
mantiene ese estado, así que nadie lo puede desincronizar. Anidar planes adentro
de planes daría lo mismo y costaría más: `aplicar` caminaría un árbol y el
colector del REPL sabría de rutas.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import reduce

ESTADOS = frozenset({"todo", "active", "done", "skipped"})

CERRADOS = frozenset({"done", "skipped"})


@dataclass(frozen=True, slots=True)
class Step:
    """Un paso: qué hay que hacer, cómo va, y qué lo cierra.

    `completes_when` nombra un predicado del registro. Vacío quiere decir que lo
    cierra el modelo, que es la válvula.

    `padre` es el id de la etapa que contiene a este paso, o vacío en la raíz.
    `exige` son los ids de los pasos de turno que hay que sembrar mientras este
    paso es la hoja activa del plan de la conversación: la exigencia viaja con
    la etapa, así que una etapa que el modelo agregue puede traer la suya, y una
    que no la traiga no exige nada.
    """

    id: str
    intent: str = ""
    status: str = "todo"
    completes_when: str = ""
    padre: str = ""
    exige: tuple[str, ...] = ()


type Plan = tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class PlanOp:
    """Una movida propuesta sobre el plan. Es lo que viaja en el canal `steps`.

    Un solo tipo para los tres verbos, con los campos que cada uno usa. Tres
    dataclasses y una unión describirían mejor lo que es y harían peor lo que hay
    que hacer con esto, que es plegarlo: `aplicar` sería un `match` sobre tipos y
    el colector del REPL tendría tres importaciones en vez de una.

    `en` es el destino de un `add_step` cuando hay más de un plan: el nombre con
    el que el host etiquetó a cada uno. `aplicar` lo ignora; lo lee el colector al
    repartir lo propuesto entre las células, y `mark` y `skip` no lo necesitan
    porque un id pertenece a un solo plan.
    """

    verb: str
    id: str = ""
    intent: str = ""
    status: str = ""
    completes_when: str = ""
    padre: str = ""
    exige: tuple[str, ...] = ()
    en: str = ""


@dataclass(frozen=True, slots=True)
class Veredicto:
    """Qué pasó con un op propuesto. El motivo es lo que el modelo lee.

    Los rechazos viajan porque un op que desaparece en silencio es un modelo que
    va a volver a escribir el mismo op el paso siguiente.
    """

    op: PlanOp
    ok: bool
    motivo: str = ""


def busca(plan: Plan, id: str) -> Step | None:
    return next((s for s in plan if s.id == id), None)


def mover(plan: Plan, id: str, status: str) -> Plan:
    """El plan con ese paso en ese estado, o el mismo plan si no hay nada que mover."""
    if status not in ESTADOS:
        return plan
    paso = busca(plan, id)
    if paso is None or paso.status == status:
        return plan
    return tuple(replace(s, status=status) if s.id == id else s for s in plan)


def aplicar(plan: Plan, op: PlanOp) -> Plan:
    """El plan después del op, o el mismo plan si el op no aplica.

    Es el único lugar donde un verbo significa algo. Un verbo desconocido, un id
    que no existe y un estado que no existe salen todos por la misma puerta:
    devolver el plan intacto. Quien tenga que distinguir "no aplica" de "aplicó"
    compara, y esa comparación es una sola línea en `admitir`.
    """
    if op.verb == "add_step":
        if not op.id or busca(plan, op.id) is not None:
            return plan
        return (*plan, Step(op.id, op.intent, "todo", op.completes_when, op.padre, op.exige))
    if op.verb == "mark":
        return mover(plan, op.id, op.status)
    if op.verb == "skip":
        return mover(plan, op.id, "skipped")
    return plan


def proyectar(steps: tuple[PlanOp, ...]) -> Plan:
    """El plan que resulta de todos los ops, en orden. Un fold, y nada más."""
    return reduce(aplicar, steps, ())


def hijos(plan: Plan, id: str) -> Plan:
    return tuple(s for s in plan if s.padre == id)


def tapado(plan: Plan, paso: Step) -> bool:
    """Si alguna etapa por encima está cerrada por su propio estado."""
    while paso.padre:
        arriba = busca(plan, paso.padre)
        if arriba is None:
            return False
        if arriba.status in CERRADOS:
            return True
        paso = arriba
    return False


def cerrado(plan: Plan, paso: Step) -> bool:
    """Si el paso está cerrado: por su estado, por el de una etapa de arriba, o por sus hijos.

    Una etapa marcada `skipped` está cerrada aunque tenga hijos abiertos, y sus
    hijos quedan cerrados con ella: saltar la facturación es saltar sus datos.
    Una etapa en `todo` con todos los hijos cerrados está cerrada igual, sin que
    nadie la marque. El estado manda cuando dice cerrado; los hijos mandan cuando
    no dice nada. Mirar hacia arriba solo por estado, y no por hijos, es lo que
    evita que el padre pregunte por el hijo y el hijo por el padre.
    """
    if paso.status in CERRADOS or tapado(plan, paso):
        return True
    abajo = hijos(plan, paso.id)
    return bool(abajo) and all(cerrado(plan, h) for h in abajo)


def activa(plan: Plan, padre: str = "") -> Step | None:
    """La hoja activa: el primer paso raíz abierto, bajando por su primer hijo abierto.

    Se busca por `padre` y no por orden del plan porque un hijo que el modelo
    agrega después va al final del canal: en orden plano, una hoja nueva de la
    primera etapa quedaría después de las hojas de la última.
    """
    for s in plan:
        if s.padre == padre and not cerrado(plan, s):
            return activa(plan, s.id) if hijos(plan, s.id) else s
    return None


def completo(plan: Plan) -> bool:
    """Si no queda nada abierto. Un plan vacío NO está completo.

    El `bool(plan)` no es defensivo, tapa un agujero medido: `all` sobre la tupla
    vacía es `True`, así que sin eso el plan recién nacido del paso 1 dice que el
    turno terminó y el agente contesta sin haber hecho nada. Es el mismo modo de
    falla que persigue `grounded`, entrando por otra puerta. Quien quiera decir
    "una checklist vacía no exige nada" pregunta por `pendientes`, no por esto.
    """
    return bool(plan) and all(cerrado(plan, s) for s in plan)


def pendientes(plan: Plan) -> Plan:
    return tuple(s for s in plan if not cerrado(plan, s))


VOCABULARIO: dict[str, frozenset[str]] = {
    "cerrado": frozenset({"mark", "skip"}),
    "acotado": frozenset({"add_step", "mark", "skip"}),
    "firme": frozenset({"add_step", "mark"}),
}
"""Qué verbos deja cada flexibilidad.

"firme" es la del plan que dura la conversación: el modelo puede agregarle
partes y marcar, y no puede saltar nada. Salió de la primera corrida de la
venta: en el turno 1, con solo la plaga por preguntar, el modelo hizo `skip` de
las otras seis hojas porque "no aplicaban todavía", y la venta quedó 6/7 antes
de empezar. Lo que todavía no toca queda abierto hasta que llegue.
"""


def deja(flex: str) -> frozenset[str]:
    """Qué verbos puede usar el modelo con esa flexibilidad.

    Una flexibilidad que no existe da el vocabulario más chico y no el más
    grande. Un nombre mal escrito en el cableado tiene que quitarle permisos al
    modelo, nunca darle los que nadie le dio.
    """
    return VOCABULARIO.get(flex, VOCABULARIO["cerrado"])


def cerrar(plan: Plan, cumplidos: frozenset[str]) -> tuple[PlanOp, ...]:
    """Los `done` que emite la máquina, uno por paso cuya compuerta ya se cumple.

    Salen como ops y no como una mutación del plan a propósito: así el plan sigue
    siendo el pliegue del canal y no una función de canal más estado del mundo. El
    día que haga falta reconstruir el plan de un turno viejo, alcanza con `steps`.
    """
    return tuple(
        PlanOp("mark", s.id, status="done")
        for s in plan
        if s.completes_when and s.completes_when in cumplidos and s.status not in CERRADOS
    )


def gateado(plan: Plan, op: PlanOp) -> str:
    """El nombre de la compuerta, si el op es un `done` que no le toca emitir al modelo.

    Un paso ya cerrado no cuenta, y el motivo es el mensaje: sobre un paso que la
    compuerta cerró hace tres pasos, decirle "lo cierro yo cuando se cumpla" es
    falso y confuso, porque ya se cumplió. Ese op cae en el chequeo de abajo y se
    rechaza por lo que realmente es, que es que no cambia nada.
    """
    if op.verb != "mark" or op.status != "done":
        return ""
    paso = busca(plan, op.id)
    if paso is None or paso.status in CERRADOS:
        return ""
    return paso.completes_when


def admitir(ops: tuple[PlanOp, ...], plan: Plan, *, flex: str) -> tuple[Veredicto, ...]:
    """Cuáles de los ops entran, juzgados en orden y contra el plan que va quedando.

    Foldea mientras admite, y eso es un reduce y no un filter. Juzgando el lote
    entero contra el plan de antes, `add_step("dosis")` seguido de
    `mark("dosis", "active")` pierde el segundo: sobre un plan sin ese paso
    `aplicar` es la identidad y el op parece un no-op. Con el plan moviéndose, el
    segundo op ve el paso que acaba de crear el primero.

    El `mark(id, "done")` sobre un paso con compuerta se rechaza siempre, incluso
    cuando la compuerta ya se cumple. Eso lo emite `cerrar`, y que haya un solo
    emisor del `done` es toda la garantía que da este módulo.
    """
    vocabulario = deja(flex)
    veredictos = []
    for op in ops:
        motivo = ""
        if op.verb not in vocabulario:
            motivo = f"`{op.verb}` no se puede usar acá (flex={flex})"
        elif compuerta := gateado(plan, op):
            motivo = f"`{op.id}` lo cierro yo cuando se cumpla `{compuerta}`"
        else:
            nuevo = aplicar(plan, op)
            if nuevo == plan:
                motivo = "no cambia nada del plan"
            else:
                plan = nuevo
        veredictos.append(Veredicto(op, not motivo, motivo))
    return tuple(veredictos)


def admitidos(veredictos: tuple[Veredicto, ...]) -> tuple[PlanOp, ...]:
    return tuple(v.op for v in veredictos if v.ok)
