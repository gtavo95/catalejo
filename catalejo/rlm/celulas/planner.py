"""La célula que admite las movidas del plan y decide si el turno puede terminar.

# Cómo veta sin poder vetar

Va después del executor y usa el mismo mecanismo que `grounded`: no puede
cancelar el DONE de nadie, y no le hace falta, porque `vote` se junta por máximo y
CONTINUE le gana a DONE. Prosa con el plan incompleto vota CONTINUE y el turno
sigue. Con el plan completo vota QUIET, y ahí el DONE del executor manda.

Dicho al revés, que es como conviene leerlo: el turno termina cuando el predicado
dice que sí, y eso se consigue LEVANTANDO el veto, no emitiendo un DONE. Por eso
esto no agrega ninguna forma nueva de terminar un turno.

# Es el plan de un turno, no de la conversación

Los ops viven en el canal `steps` del Log, y el Log arranca limpio en cada
pregunta (`agro.responder`). La semilla se vuelve a emitir entera, con todo en
`[ ]`, y el plan se cierra adentro del `loop` de esa pregunta o no se cierra.
Nada de lo que cerró la pregunta anterior cuenta para esta, por la misma razón
por la que `grounded` no arrastra `reads`: un `cito_la_fuente` cumplido en la
pregunta uno cerraría `fuente` en la cinco sin que la cinco cite nada.

Un plan que atraviese turnos usaría el mismo fold: los `steps` con los que salió
un turno son la semilla del siguiente, y por la ley del fold el plan de la
conversación es el pliegue de todos. Lo que no puede reusar es el veto de abajo:
un plan de conversación abierto al final de un turno es lo normal, no un turno
mal terminado. Eso hoy no existe.

# Qué cuesta

Un mensaje `[plan] estado:` por paso en el que el plan se movió. Los pasos en los
que nadie lo tocó no dicen nada, así que el costo es proporcional a la actividad
del plan y no al largo del turno. La contra está medida a medias: `window` conserva
`said[0]` y los últimos seis, así que en una racha larga de greps la checklist se
puede caer de la ventana. La alternativa es un mensaje por paso siempre, y eso es
una fila de la bitácora, no una decisión de diseño.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

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
    Veredicto,
    admitidos,
    admitir,
    cerrar,
    completo,
    pendientes,
    proyectar,
)

from ..repl import Verbos
from .executor import extract_code

type Registro = Mapping[str, Callable[[Log], bool]]

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


def render_plan(plan: Plan) -> str:
    """La checklist como la lee el modelo.

    Un solo renderizador, y lo usa también el host para meter la semilla en el
    turno de apertura: si el plan arranca dibujado de una forma en el pedido y de
    otra en los avisos, el modelo cree que son dos cosas.
    """
    lineas = [ESTADO]
    for s in plan:
        intent = f": {s.intent}" if s.intent else ""
        cola = (
            f" (lo cierro yo con `{s.completes_when}`)"
            if s.completes_when and s.status not in CERRADOS
            else ""
        )
        lineas.append(f"{MARCAS.get(s.status, '[?]')} {s.id}{intent}{cola}")
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
    """Si el último dicho es una respuesta final y no un pedido de ejecución.

    Comparte `extract_code` con el executor y con `grounded`: un solo parser
    decide qué cuenta como código en todo el repo.
    """
    if not seen.said:
        return False
    ultimo = seen.said[-1]
    return ultimo.role is Role.ASSISTANT and not extract_code(ultimo.text)


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


def planner(
    verbos: Verbos,
    registro: Registro,
    *,
    flex: str = "acotado",
    semilla: tuple[PlanOp, ...] = (),
) -> Cell:
    """Drena lo que el modelo propuso, admite lo que corresponde y cierra lo que ya está.

    `verbos` es el colector que llenan los builtins `add_step`, `mark` y `skip`
    desde el REPL. Tiene que ser la misma instancia cuyos `builtins` entraron al
    `extra` del workspace, porque esta célula lo drena en cada paso.

    `registro` mapea nombres a predicados sobre el Log. Un paso con
    `completes_when="grep"` se cierra cuando `registro["grep"](seen)` da
    verdadero, y lo cierra esta célula, nunca el modelo. Un nombre que no está en
    el registro es un `Fail`, porque ese paso no se cerraría nunca.

    `flex` es qué verbos puede usar el modelo: "cerrado" deja `mark` y `skip`,
    "acotado" suma `add_step`. Un valor desconocido cae en "cerrado".

    `semilla` son los ops con los que arranca el plan cuando el canal `steps` está
    vacío: la checklist que el host define de antemano. Sin semilla el plan
    arranca vacío y solo crece si `flex` deja `add_step`.

    El orden adentro del paso importa y es este: primero la semilla si el canal
    está vacío, después lo que propuso el modelo juzgado contra ese plan, y recién
    al final las compuertas contra el plan ya movido. Al revés, un paso que el
    modelo crea y cuya compuerta ya se cumple nace abierto y se cierra un paso
    después, gratis pero tarde.

    La semilla la emite esta célula y no el host, para que el que llama no tenga
    que armar un Log con canales adentro. El host la nombra dos veces igual: acá y
    en el turno de apertura, con el mismo `render_plan`.
    """

    async def cell(seen: Log) -> Log:
        arranque = () if seen.steps else semilla
        veredictos = admitir(verbos.tomar(), proyectar(seen.steps + arranque), flex=flex)
        movidas = arranque + admitidos(veredictos)
        listos, fallas = cumplidos(registro, seen)
        plan = proyectar(seen.steps + movidas)
        movidas += cerrar(plan, listos)
        plan = proyectar(seen.steps + movidas)
        fallas += huerfanos(plan, registro)

        if contesto(seen) and not completo(plan):
            if avisado(seen):
                return Log(steps=movidas, fails=fallas + (INCOMPLETO,))
            dicho = Message(Role.USER, f"{render_plan(plan)}\n\n{render_aviso(plan)}")
            return Log(said=(dicho,), steps=movidas, fails=fallas, vote=Status.CONTINUE)

        rechazos = [f"rechazado {render_op(v.op)}: {v.motivo}" for v in veredictos if not v.ok]
        if not movidas and not rechazos:
            return Log(fails=fallas) if fallas else ZERO
        if contesto(seen):
            return Log(steps=movidas, fails=fallas)
        cuerpo = "\n".join([render_plan(plan), *rechazos])
        return Log(said=(Message(Role.USER, cuerpo),), steps=movidas, fails=fallas)

    return cell
