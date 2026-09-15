"""Las células del plan: mover un plan, exigirle al turno, y derivar uno del otro.

# Cómo veta sin poder vetar

`exigir` va después del executor y usa el mismo mecanismo que `grounded`: no
puede cancelar el DONE de nadie, y no le hace falta, porque `vote` se junta por
máximo y CONTINUE le gana a DONE. Prosa con la checklist abierta vota CONTINUE y
el turno sigue. Con la checklist cerrada vota QUIET, y ahí el DONE del executor
manda.

Dicho al revés, que es como conviene leerlo: el turno termina cuando el predicado
dice que sí, y eso se consigue LEVANTANDO el veto, no emitiendo un DONE. Por eso
esto no agrega ninguna forma nueva de terminar un turno.

# Dos células, porque el veto es del turno y el fold no

`avanzar` mueve un plan: emite la semilla si el canal está vacío, admite lo que
el modelo propuso, cierra por compuerta y dibuja. Nunca vota. `exigir` mira la
checklist de `steps` y veta. Son dos porque sirven a dos planes distintos y solo
uno de los dos se exige: el plan de la conversación, en el canal `course`, puede
quedar abierto al final de un turno y eso es lo normal, no un turno mal
terminado. `planner` es `then(avanzar, exigir)`, el planner del turno de siempre.

`avanzar` es la misma célula para los dos canales. Lo que cambia es de cuál
proyecta y a cuál emite, y qué parte del colector toma: la de la conversación
toma lo que lleva su etiqueta o un id de su plan, y la del turno, que va última,
toma el resto.

# La exigencia viaja con la etapa

`derivar` es la costura entre los dos planes. Mira la hoja activa de `course` y
siembra en `steps` los pasos de turno que esa hoja declara en `exige`, sacados
del catálogo del host. Con la hoja `plaga` activa, `exige` es vacío y el turno
no tiene checklist: el modelo puede preguntar qué plaga es sin saltar nada. Con
`producto` activa, el turno exige buscar y citar. Y cuando `producto` se cierra
a mitad del turno, `derivar` ve a `dosis` activa y la agrega a la checklist: la
dosis se exige en el turno en que aparece el producto, no en el siguiente.

Es idempotente gratis: `aplicar` devuelve el mismo plan ante un `add_step` de un
id que ya existe, y `derivar` filtra antes para no ensuciar el canal con no-ops.

# Qué cuesta

Un mensaje `[plan] estado:` por paso en el que la checklist se movió. Los pasos
en los que nadie la tocó no dicen nada, así que el costo es proporcional a la
actividad del plan y no al largo del turno. La contra está medida a medias:
`window` conserva `said[0]` y los últimos seis, así que en una racha larga de
greps la checklist se puede caer de la ventana. El plan de la conversación va
con `dibuja=False` en `agro.py`: el modelo lo ve una vez por turno, en el
pedido, y de la célula solo le llegan los rechazos.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal

from catalejo.core import (
    CERRADOS,
    ZERO,
    Cell,
    Fail,
    Log,
    Message,
    Plan,
    PlanOp,
    Role,
    Status,
    Step,
    activa,
    admitidos,
    admitir,
    aplicar,
    busca,
    cerrado,
    cerrar,
    hijos,
    pendientes,
    proyectar,
    then,
)

from ..repl import Verbos
from .executor import extract_code

type Registro = Mapping[str, Callable[[Log], bool]]

type Canal = Literal["steps", "course"]
"""Los dos canales de movidas: el del turno y el de la conversación."""

ESTADO = "[plan] estado:"

FALTAN = "[plan] faltan pasos:"

INCOMPLETO = Fail("plan", "contestó con pasos abiertos")

MARCAS = {"todo": "[ ]", "active": "[>]", "done": "[x]", "skipped": "[-]"}


def render_op(op: PlanOp) -> str:
    """El op como el modelo lo escribió, para que reconozca cuál le rechacé."""
    args = [f'"{op.id}"']
    if op.verb == "add_step" and op.intent:
        args.append(f'"{op.intent}"')
    if op.verb == "mark":
        args.append(f'"{op.status}"')
    return f"{op.verb}({', '.join(args)})"


def render_paso(plan: Plan, s: Step, sangria: str) -> str:
    marca = MARCAS.get(s.status, "[?]")
    if s.status not in CERRADOS and cerrado(plan, s):
        marca = MARCAS["done"]
    intent = f": {s.intent}" if s.intent else ""
    cola = (
        f" (lo cierro yo con `{s.completes_when}`)"
        if s.completes_when and not cerrado(plan, s)
        else ""
    )
    return f"{sangria}{marca} {s.id}{intent}{cola}"


def render_plan(plan: Plan, titulo: str = ESTADO, padre: str = "", sangria: str = "") -> str:
    """La checklist como la lee el modelo, con las etapas sangradas.

    Un solo renderizador, y lo usa también el host para meter la semilla en el
    turno de apertura: si el plan arranca dibujado de una forma en el pedido y de
    otra en los avisos, el modelo cree que son dos cosas. `titulo` es lo único
    que cambia entre el plan del turno y el de la conversación.

    Una etapa cerrada por sus hijos se dibuja `[x]` aunque su estado diga `todo`:
    lo que se dibuja es lo que `cerrado` calcula, no el campo.
    """
    lineas = [titulo] if titulo else []
    for s in plan:
        if s.padre != padre:
            continue
        lineas.append(render_paso(plan, s, sangria))
        if hijos(plan, s.id):
            lineas.append(render_plan(plan, "", s.id, sangria + "    "))
    return "\n".join(lineas)


def render_aviso(plan: Plan) -> str:
    """El único empujón. Dice qué falta y qué hacer con cada cosa que falta."""
    abiertos = pendientes(plan)
    nombres = ", ".join(f"`{s.id}`" for s in abiertos)
    conmigo = [s for s in abiertos if s.completes_when]
    tuyos = [s for s in abiertos if not s.completes_when]
    lineas = [f"{FALTAN} {nombres}."]
    if tuyos:
        cierres = "; ".join(f'mark("{s.id}", "done")' for s in tuyos)
        lineas.append(
            f"Si ya los hiciste, cerralos vos en un bloque de código: {cierres}. "
            f"Si alguno no aplica, `skip(\"id\")` y explicá por qué en la respuesta."
        )
    if conmigo:
        faltantes = "; ".join(f"`{s.id}` espera `{s.completes_when}`" for s in conmigo)
        lineas.append(f"Estos los cierro yo cuando pase el hecho: {faltantes}.")
    lineas.append("Después de eso, escribí la respuesta final.")
    return "\n".join(lineas)


def avisado(seen: Log) -> bool:
    """Si el empujón ya salió en este turno.

    Busca la marca adentro del texto, al contrario de `grounded`, que compara el
    mensaje entero. Dos razones: el aviso nombra los pasos que faltan, así que el
    texto cambia entre un paso y el siguiente, y encima viaja pegado al estado del
    plan en el mismo mensaje. Con igualdad, un plan que se movió en el medio se
    lleva un aviso nuevo y el tope de "avisa una vez" no existe.

    Solo mira los mensajes del usuario, que son los que escribe el host. Un modelo
    que copie la marca en su respuesta no puede apagarse el aviso a sí mismo.
    """
    return any(m.role is Role.USER and FALTAN in m.text for m in seen.said)


def contesto(seen: Log) -> bool:
    """Si lo último que dijo el modelo es una respuesta final y no un pedido de ejecución.

    Mira el último dicho del MODELO y no el último dicho a secas, porque en el
    mismo paso, después de la respuesta, otras células agregan mensajes del
    usuario: el aviso de `grounded`, la checklist que `derivar` acaba de sembrar.
    El modelo habla una vez por paso, así que todo lo que hay después de su
    último mensaje es de este paso, y la pregunta sigue siendo si contestó.

    Comparte `extract_code` con el executor y con `grounded`: un solo parser
    decide qué cuenta como código en todo el repo.
    """
    ultimo = next((m for m in reversed(seen.said) if m.role is Role.ASSISTANT), None)
    return ultimo is not None and not extract_code(ultimo.text)


def cumplidos(registro: Registro, seen: Log) -> tuple[frozenset[str], tuple[Fail, ...]]:
    """Qué compuertas se cumplen ahora mismo, y cuáles reventaron al preguntar.

    Un predicado es código del host y puede tener un bug. Si levanta, cuenta como
    no cumplido y queda anotado: una célula no propaga excepciones, y una compuerta
    que revienta en silencio es un requisito que desaparece.
    """
    listos: set[str] = set()
    fallas: list[Fail] = []
    for nombre, predicado in registro.items():
        try:
            if predicado(seen):
                listos.add(nombre)
        except Exception as e:
            fallas.append(Fail("plan", f"el predicado `{nombre}` reventó: {type(e).__name__}: {e}"))
    return frozenset(listos), tuple(fallas)


def huerfanos(plan: Plan, registro: Registro) -> tuple[Fail, ...]:
    """Los pasos cuya compuerta no existe en el registro.

    Cierra por el lado seguro: ese paso no se cierra nunca, así que el plan no se
    completa, así que el turno sale con `Fail` en vez de salir como si estuviera
    todo bien. Al revés, un nombre mal escrito desactivaría el requisito y nadie
    se enteraría. Se repite en cada paso y no importa: `fails` es un conjunto.
    """
    return tuple(
        Fail(
            "plan",
            f"`{s.completes_when}` no está en el registro, así que `{s.id}` no se cierra nunca",
        )
        for s in plan
        if s.completes_when and s.completes_when not in registro
    )


def ops_de(seen: Log, canal: Canal) -> tuple[PlanOp, ...]:
    """Las movidas de ese canal."""
    return seen.steps if canal == "steps" else seen.course


def con_ops(log: Log, canal: Canal, ops: tuple[PlanOp, ...]) -> Log:
    """El mismo Log con esas movidas en ese canal."""
    return replace(log, steps=ops) if canal == "steps" else replace(log, course=ops)


def avanzar(
    verbos: Verbos,
    registro: Registro,
    *,
    canal: Canal = "steps",
    flex: str = "acotado",
    semilla: tuple[PlanOp, ...] = (),
    etiqueta: str = "",
    titulo: str = ESTADO,
    dibuja: bool = True,
) -> Cell:
    """Mueve el plan de un canal: semilla, lo que el modelo propuso, y los cierres por compuerta.

    `verbos` es el colector que llenan los builtins `add_step`, `mark` y `skip`
    desde el REPL. Tiene que ser la misma instancia cuyos `builtins` entraron al
    `extra` del workspace, porque esta célula lo drena en cada paso.

    `registro` mapea nombres a predicados sobre el Log. Un paso con
    `completes_when="grep"` se cierra cuando `registro["grep"](seen)` da
    verdadero, y lo cierra esta célula, nunca el modelo. Un nombre que no está en
    el registro es un `Fail`, porque ese paso no se cerraría nunca.

    `canal` es de dónde proyecta y adónde emite: `steps` para el turno, `course`
    para la conversación. `etiqueta` es cómo el modelo nombra a este plan en
    `add_step(..., en=...)`; con etiqueta, la célula toma del colector todo op
    cuyo id ya está en su plan, y los `add_step` que la llevan o cuyo `padre`
    está en su plan, y deja el resto. Sin etiqueta toma todo lo que quedó, así
    que la célula sin etiqueta va última. Las dos reglas de más salieron de las
    primeras corridas: el modelo escribió `en="diagnostico"`, el id de la etapa,
    y esa parte de la venta cayó en la checklist del turno; y escribió
    `add_step("plaga", ...)` sin nada, y el id de una hoja de la venta apareció
    como paso del turno. Un id pertenece a un solo plan, también al nacer. Y el
    reparto pliega mientras juzga, como `admitir`: el `mark` que sigue al
    `add_step` de su propio id en el mismo bloque es del mismo plan, y sin el
    fold caía en el otro y volvía rechazado como "no cambia nada".

    `dibuja=False` no dibuja el estado, solo los rechazos. Es para el plan de la
    conversación, que el modelo ve una vez por turno en el pedido: dibujarlo en
    cada paso que se mueve es prompt que no compra nada, y callar los rechazos
    es dejar al modelo repitiendo el mismo op.

    `flex` es qué verbos puede usar el modelo: "cerrado" deja `mark` y `skip`,
    "acotado" suma `add_step`. Un valor desconocido cae en "cerrado".

    `semilla` son los ops con los que arranca el plan cuando el canal está
    vacío: la checklist que el host define de antemano. Sin semilla el plan
    arranca vacío y solo crece si `flex` deja `add_step`, o si otra célula
    siembra en el canal, que es lo que hace `derivar`. Un canal que el host
    sembró desde afuera no está vacío, y la semilla no se vuelve a emitir: así
    el plan de la conversación entra con su historia en cada turno.

    El orden adentro del paso importa y es este: primero la semilla si el canal
    está vacío, después lo que propuso el modelo juzgado contra ese plan, y recién
    al final las compuertas contra el plan ya movido. Al revés, un paso que el
    modelo crea y cuya compuerta ya se cumple nace abierto y se cierra un paso
    después, gratis pero tarde.

    No vota nunca. Con el modelo ya contestado (prosa) emite las movidas sin
    dibujar: dibujar ahí es hablarle a alguien que ya terminó, y si hace falta
    vetar, eso lo hace `exigir` con su propio mensaje.
    """

    def toma(plan: Plan) -> Callable[[PlanOp], bool] | None:
        if not etiqueta:
            return None
        visto = plan

        def propio(op: PlanOp) -> bool:
            nonlocal visto
            mio = busca(visto, op.id) is not None or (
                op.verb == "add_step"
                and (op.en == etiqueta or (bool(op.padre) and busca(visto, op.padre) is not None))
            )
            if mio:
                visto = aplicar(visto, op)
            return mio

        return propio

    async def cell(seen: Log) -> Log:
        previos = ops_de(seen, canal)
        arranque = () if previos else semilla
        plan = proyectar(previos + arranque)
        veredictos = admitir(verbos.tomar(toma(plan)), plan, flex=flex)
        movidas = arranque + admitidos(veredictos)
        listos, fallas = cumplidos(registro, seen)
        plan = proyectar(previos + movidas)
        movidas += cerrar(plan, listos)
        plan = proyectar(previos + movidas)
        fallas += huerfanos(plan, registro)

        rechazos = [f"rechazado {render_op(v.op)}: {v.motivo}" for v in veredictos if not v.ok]
        if not movidas and not rechazos:
            return Log(fails=fallas) if fallas else ZERO
        if contesto(seen) or (not dibuja and not rechazos):
            return con_ops(Log(fails=fallas), canal, movidas)
        cuerpo = "\n".join([render_plan(plan, titulo), *rechazos] if dibuja else rechazos)
        return con_ops(Log(said=(Message(Role.USER, cuerpo),), fails=fallas), canal, movidas)

    return cell


def exigir() -> Cell:
    """Veta la respuesta final mientras la checklist del turno tenga pasos abiertos.

    Mira `steps` y el último mensaje. Si el modelo contestó en prosa con
    pendientes, vota CONTINUE y le escribe qué falta. Si vuelve a contestar con
    pendientes, lo deja terminar y anota `INCOMPLETO`. Con la checklist cerrada
    vota QUIET. No recibe el plan porque adentro de `then` ve `merge(seen, acc)`,
    y ahí ya están los ops que `avanzar` acaba de emitir en este mismo paso.

    Veta por `pendientes` y no por `not completo` a propósito. Para una
    checklist con pasos es lo mismo; para la vacía es la diferencia entre "este
    turno no exige nada" y vetar toda respuesta. La checklist vacía existe: es la
    del turno en que la hoja activa de la conversación no exige nada.
    """

    async def cell(seen: Log) -> Log:
        plan = proyectar(seen.steps)
        if not contesto(seen) or not pendientes(plan):
            return ZERO
        if avisado(seen):
            return Log(fails=(INCOMPLETO,))
        dicho = Message(Role.USER, f"{render_plan(plan)}\n\n{render_aviso(plan)}")
        return Log(said=(dicho,), vote=Status.CONTINUE)

    return cell


def derivar(
    pasos: Mapping[str, PlanOp],
    *,
    desde: Canal = "course",
    hacia: Canal = "steps",
    titulo: str = ESTADO,
) -> Cell:
    """Siembra en la checklist del turno lo que exige la hoja activa de la conversación.

    `pasos` es el catálogo del host: los pasos de turno por id, como `add_step`.
    La hoja activa de `desde` declara en `exige` cuáles de esos ids valen mientras
    ella esté activa, y esta célula los emite a `hacia`, salvo los que ya están
    en ese plan, abiertos o cerrados: un paso que el modelo saltó tampoco vuelve.
    Un id que no está en el catálogo es un `Fail`, por lo mismo que una
    compuerta que no está en el registro: un requisito mal escrito tiene que
    verse, no desaparecer.

    Dibuja la checklist cuando siembra algo, porque el `avanzar` de `hacia` corre
    después y solo dibuja lo que él mismo movió: sin esto, el modelo no vería la
    checklist hasta que algo más la toque. Con el modelo ya contestado no dibuja,
    por lo mismo que `avanzar`: si lo sembrado queda abierto, `exigir` lo dibuja
    con su aviso, y si no, no hay a quién mostrárselo.
    """

    async def cell(seen: Log) -> Log:
        hoja = activa(proyectar(ops_de(seen, desde)))
        if hoja is None or not hoja.exige:
            return ZERO
        previos = ops_de(seen, hacia)
        checklist = proyectar(previos)
        nuevos = tuple(pasos[i] for i in hoja.exige if i in pasos and busca(checklist, i) is None)
        fallas = tuple(
            Fail("plan", f"`{i}` no está en el catálogo de pasos, así que `{hoja.id}` lo exige en vano")
            for i in hoja.exige
            if i not in pasos
        )
        if not nuevos:
            return Log(fails=fallas) if fallas else ZERO
        if contesto(seen):
            return con_ops(Log(fails=fallas), hacia, nuevos)
        dicho = Message(Role.USER, render_plan(proyectar(previos + nuevos), titulo))
        return con_ops(Log(said=(dicho,), fails=fallas), hacia, nuevos)

    return cell


def planner(
    verbos: Verbos,
    registro: Registro,
    *,
    flex: str = "acotado",
    semilla: tuple[PlanOp, ...] = (),
) -> Cell:
    """El planner del turno: `then(avanzar, exigir)` sobre `steps`.

    El orden importa: `exigir` proyecta el plan que `avanzar` acaba de mover en
    el mismo paso, así que un cierre por compuerta en este paso levanta el veto
    en este paso. Al revés, el modelo contestaría bien y se llevaría un aviso por
    un paso que ya estaba cerrado.

    La semilla la emite `avanzar` y no el host, para que el que llama no tenga
    que armar un Log con canales adentro. El host la nombra dos veces igual: acá y
    en el turno de apertura, con el mismo `render_plan`.
    """
    return then(avanzar(verbos, registro, flex=flex, semilla=semilla), exigir())
